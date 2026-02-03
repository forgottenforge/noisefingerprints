#!/usr/bin/env python3
"""Compare variance scaling alpha across three randomness sources:
  1. PRNG  (numpy PCG64 — deterministic, perfect i.i.d.)
  2. HRNG  (os.urandom / Intel RDRAND — thermal noise in CPU)
  3. QPU   (Rigetti Ankaa-3 and IonQ Forte-1 — quantum noise)

Expectation: PRNG -> alpha = -1.0 exactly
             HRNG -> alpha ~ -1.0 (thermal noise, mostly i.i.d.)
             QPU  -> alpha != -1.0 (hardware-dependent correlations)
"""

import json
import os
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter, find_peaks, peak_prominences
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- QPU data files ---
RIGETTI_FILE = "data/vacuum_telescope_v1/replication_rigetti_qpu_20260201_100409.json"
IONQ_FILE = "data/vacuum_telescope_v1/replication_ionq_qpu_20260201_095758.json"
OUT_FILE = "data/vacuum_telescope_v1/figures/rng_comparison.png"

N_BITS = 6          # match QPU qubit count
N_SHOTS = 10000     # match Rigetti boost count


# ============================================================================
# Bitstring generators
# ============================================================================

def generate_prng_bitstrings(n_shots, n_bits, seed=42):
    """Generate bitstrings from numpy PRNG (PCG64, deterministic)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_shots, n_bits))
    return [''.join(str(b) for b in row) for row in bits]


def generate_hrng_bitstrings(n_shots, n_bits):
    """Generate bitstrings from OS entropy (RDRAND/hardware RNG)."""
    bitstrings = []
    for _ in range(n_shots):
        byte = os.urandom(1)[0]
        bits = format(byte, '08b')[-n_bits:]  # take last n_bits
        bitstrings.append(bits)
    return bitstrings


def load_qpu_bitstrings(filepath, block_name, gamma_target):
    """Load bitstrings from QPU JSON data."""
    with open(filepath) as f:
        data = json.load(f)
    block = data['blocks'][block_name]
    for m in block['measurements']:
        if abs(m['gamma'] - gamma_target) < 0.01:
            return m['bitstrings']
    return None


# ============================================================================
# Analysis (same as PRA paper)
# ============================================================================

def hamming_weights(bitstrings):
    return np.array([sum(int(b) for b in bs) for bs in bitstrings])


def variance_scaling(bitstrings):
    """Compute alpha from inter-batch variance scaling."""
    n = len(bitstrings)
    batch_sizes = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
    batch_sizes = [k for k in batch_sizes if k <= n // 3]

    ks, variances = [], []
    for K in batch_sizes:
        nb = n // K
        vals = [float(np.var(hamming_weights(bitstrings[b*K:(b+1)*K]))) for b in range(nb)]
        if len(vals) >= 3:
            ks.append(K)
            variances.append(np.var(vals))

    ks = np.array(ks, dtype=float)
    variances = np.array(variances)
    valid = variances > 0

    log_k = np.log(ks[valid])
    log_v = np.log(variances[valid] + 1e-30)

    slope, intercept, r, p, se = stats.linregress(log_k, log_v)
    return ks[valid], variances[valid], log_k, log_v, slope, se, intercept


def compute_sigma_c(sigmas, observables, smoothing_window=5):
    """Batch-size susceptibility (same as vacuum_statistics.py)."""
    if len(observables) < smoothing_window:
        obs_smooth = observables
    else:
        polyorder = min(3, smoothing_window - 1)
        obs_smooth = savgol_filter(observables, smoothing_window, polyorder, mode='nearest')
    chi = np.abs(np.gradient(obs_smooth, sigmas))
    peaks, _ = find_peaks(chi, prominence=0.001)
    if len(peaks) == 0:
        peak_idx = np.argmax(chi)
        sigma_c = sigmas[peak_idx]
        baseline = np.median(chi) + 1e-15
        kappa = chi[peak_idx] / baseline
    else:
        prominences = peak_prominences(chi, peaks)[0]
        best_peak_idx = peaks[np.argmax(prominences)]
        sigma_c = sigmas[best_peak_idx]
        baseline = np.median(chi) + 1e-15
        kappa = prominences.max() / baseline
    return chi, float(sigma_c), float(kappa)


def batch_analysis(bitstrings):
    """Full batch analysis: variance scaling + susceptibility."""
    n = len(bitstrings)
    batch_sizes = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
    batch_sizes = [k for k in batch_sizes if k <= n // 3]

    obs_func = lambda bs: float(np.var(hamming_weights(bs)))
    batch_ks, batch_means = [], []
    for K in batch_sizes:
        nb = n // K
        vals = [obs_func(bitstrings[b*K:(b+1)*K]) for b in range(nb)]
        if len(vals) >= 3:
            batch_ks.append(K)
            batch_means.append(float(np.mean(vals)))

    batch_ks = np.array(batch_ks, dtype=float)
    batch_means = np.array(batch_means)
    chi, sigma_c, kappa = compute_sigma_c(batch_ks, batch_means)
    return batch_ks, chi, sigma_c, kappa


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  RNG COMPARISON: PRNG vs HRNG vs QPU")
    print("=" * 70)

    # --- Generate / load data ---
    sources = {}

    # PRNG
    prng_bs = generate_prng_bitstrings(N_SHOTS, N_BITS)
    sources['PRNG\n(numpy PCG64)'] = {'bs': prng_bs, 'color': '#888888', 'marker': 'D'}
    print(f"\n  PRNG: {len(prng_bs)} bitstrings generated (numpy PCG64, seed=42)")

    # HRNG
    hrng_bs = generate_hrng_bitstrings(N_SHOTS, N_BITS)
    sources['HRNG\n(os.urandom)'] = {'bs': hrng_bs, 'color': '#2ca02c', 'marker': '^'}
    print(f"  HRNG: {len(hrng_bs)} bitstrings generated (os.urandom)")

    # Rigetti QPU
    rig_bs = load_qpu_bitstrings(RIGETTI_FILE, "rigetti_boost", 0.0)
    if rig_bs:
        sources['Rigetti\nAnkaa-3'] = {'bs': rig_bs, 'color': '#1f77b4', 'marker': 'o'}
        print(f"  Rigetti: {len(rig_bs)} bitstrings loaded (gamma=0)")

    # IonQ QPU
    ionq_bs = load_qpu_bitstrings(IONQ_FILE, "R23_batch", 0.0)
    if ionq_bs:
        sources['IonQ\nForte-1'] = {'bs': ionq_bs, 'color': '#ff7f0e', 'marker': 's'}
        print(f"  IonQ: {len(ionq_bs)} bitstrings loaded (gamma=0)")

    # --- Analyze all ---
    print(f"\n{'Source':<20} {'N':>6} {'alpha':>8} {'SE':>6} {'p(H0)':>10} {'sigma_c':>8} {'kappa':>8}")
    print("-" * 70)

    results = {}
    for name, src in sources.items():
        bs = src['bs']
        ks, var, log_k, log_v, alpha, se, intercept = variance_scaling(bs)
        batch_ks, chi, sigma_c, kappa = batch_analysis(bs)

        # p-value for H0: alpha = -1
        t_stat = (alpha - (-1.0)) / se
        df = len(log_k) - 2
        p_val = 2 * stats.t.sf(abs(t_stat), df)

        label = name.replace('\n', ' ')
        print(f"  {label:<18} {len(bs):>6} {alpha:>+8.3f} {se:>6.3f} {p_val:>10.2e} {sigma_c:>8.0f} {kappa:>8.2f}")

        results[name] = {
            'log_k': log_k, 'log_v': log_v, 'alpha': alpha, 'se': se,
            'intercept': intercept, 'p': p_val,
            'batch_ks': batch_ks, 'chi': chi, 'sigma_c': sigma_c, 'kappa': kappa,
            'color': src['color'], 'marker': src['marker'], 'n': len(bs)
        }

    # ========================================================================
    # Plot
    # ========================================================================
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel A: Variance scaling log-log ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('(a)  Variance scaling: $\\ln V(K)$ vs $\\ln K$', fontsize=13, fontweight='bold')

    for name, r in results.items():
        ax1.plot(r['log_k'], r['log_v'], r['marker'], color=r['color'], ms=7, alpha=0.8,
                 label=f"{name.replace(chr(10), ' ')}:  $\\alpha = {r['alpha']:+.3f} \\pm {r['se']:.3f}$")
        fit_x = np.linspace(r['log_k'].min(), r['log_k'].max(), 50)
        ax1.plot(fit_x, r['alpha'] * fit_x + r['intercept'], '-', color=r['color'], lw=1.5, alpha=0.5)

    # i.i.d. reference
    x_ref = np.array([1.5, 7.0])
    y_mid = np.mean([r['log_v'].mean() for r in results.values()])
    x_mid = np.mean([r['log_k'].mean() for r in results.values()])
    iid_int = y_mid - (-1.0) * x_mid
    ax1.plot(x_ref, -1.0 * x_ref + iid_int, 'k--', lw=2, alpha=0.4,
             label=r'i.i.d. prediction: $\alpha = -1.0$')

    ax1.set_xlabel(r'$\ln(K)$', fontsize=12)
    ax1.set_ylabel(r'$\ln(\mathrm{Var})$', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.2)

    # --- Panel B: Susceptibility chi(K) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title(r'(b)  Susceptibility $\chi(K)$', fontsize=13, fontweight='bold')

    for name, r in results.items():
        chi_norm = r['chi'] / (r['chi'].max() + 1e-30)  # normalize for comparison
        ax2.plot(r['batch_ks'], chi_norm, f"{r['marker']}-", color=r['color'],
                 ms=5, lw=1.5, alpha=0.8, label=name.replace('\n', ' '))
        # mark sigma_c
        sc_idx = np.argmin(np.abs(r['batch_ks'] - r['sigma_c']))
        ax2.plot(r['sigma_c'], chi_norm[sc_idx], '*', color=r['color'], ms=14, zorder=5)

    ax2.set_xlabel('Batch size $K$', fontsize=12)
    ax2.set_ylabel(r'$\chi(K)$ (normalized)', fontsize=12)
    ax2.set_xscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # --- Panel C: Summary bar chart ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title(r'(c)  Scaling exponent $\alpha$ summary', fontsize=13, fontweight='bold')

    names_short = []
    alphas = []
    ses = []
    colors = []
    for name, r in results.items():
        names_short.append(name.replace('\n', '\n'))
        alphas.append(r['alpha'])
        ses.append(r['se'])
        colors.append(r['color'])

    x_pos = np.arange(len(names_short))
    bars = ax3.bar(x_pos, alphas, yerr=ses, capsize=5, color=colors, alpha=0.7, edgecolor='black', lw=0.8)
    ax3.axhline(-1.0, color='red', ls='--', lw=2, label=r'i.i.d.: $\alpha = -1.0$')
    ax3.axhline(0.0, color='black', ls='-', lw=0.5, alpha=0.3)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names_short, fontsize=9, ha='center')
    ax3.set_ylabel(r'$\alpha$', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.2, axis='y')

    # Add p-values as text
    for i, (name, r) in enumerate(results.items()):
        p = r['p']
        if p < 0.001:
            p_str = f"p < 0.001"
        elif p < 0.05:
            p_str = f"p = {p:.3f}"
        else:
            p_str = f"p = {p:.2f}"
        y_offset = 0.08 if alphas[i] >= 0 else -0.15
        ax3.text(i, alphas[i] + y_offset, p_str, ha='center', va='bottom', fontsize=8, fontstyle='italic')

    fig.suptitle('Randomness Source Comparison: PRNG vs Hardware RNG vs Quantum Processors',
                 fontsize=14, fontweight='bold', y=0.98)

    fig.savefig(OUT_FILE, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {OUT_FILE}")

    # --- Interpretation ---
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    for name, r in results.items():
        label = name.replace('\n', ' ')
        if r['p'] > 0.05:
            print(f"  {label}: alpha = {r['alpha']:+.3f} -- CONSISTENT with i.i.d. (p = {r['p']:.3f})")
        else:
            print(f"  {label}: alpha = {r['alpha']:+.3f} -- DEVIATES from i.i.d. (p = {r['p']:.2e})")


if __name__ == "__main__":
    main()
