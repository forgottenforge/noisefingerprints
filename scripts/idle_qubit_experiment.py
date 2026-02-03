#!/usr/bin/env python3
"""Idle Qubit Noise Isolation Experiment.

Systematic comparison of 4 circuit types to isolate noise sources:
  C1: Idle |0>^6       — readout error + T1 only
  C2: Hadamard |+>^6   — + T2 dephasing (no entanglement)
  C3: Bell pairs        — + local CNOT noise (pairwise entanglement)
  C4: Full chain        — + global entanglement noise (existing PRA data)

Analysis: variance scaling (alpha), qubit-pair correlations, temporal
autocorrelation, shuffled-shots control, PRNG baseline.

Usage:
  python idle_qubit_experiment.py --mode sim    # Test on simulator (free)
  python idle_qubit_experiment.py --mode qpu    # Run on Rigetti ($11.40)
  python idle_qubit_experiment.py --analyze FILE # Analyze existing data
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter, find_peaks, peak_prominences
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ============================================================================
# Constants
# ============================================================================

N_QUBITS = 6
SHOTS = 10000
N_SHUFFLE = 1000
BATCH_SIZES = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000]

# Braket
RIGETTI_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
COST_PER_TASK = 0.30
COST_PER_SHOT = 0.00035
EUR_PER_USD = 1 / 1.08

# Existing PRA data (C4: full chain)
CHAIN_DATA_FILE = "data/vacuum_telescope_v1/replication_rigetti_qpu_20260201_100409.json"

# Output
OUT_DIR = Path("data/vacuum_telescope_v1")
OUT_DATA = OUT_DIR / "idle_qubit_experiment.json"
FIG_DIR = OUT_DIR / "figures"

# Qubit adjacency on Rigetti Ankaa-3 (chain 0-1-2-3-4-5)
QUBIT_PAIRS = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS)]
PAIR_DISTANCES = {(i, j): abs(j - i) for i, j in QUBIT_PAIRS}

# Circuit colors
COLORS = {
    'idle': '#7f7f7f',
    'hadamard': '#2ca02c',
    'bell': '#9467bd',
    'chain': '#1f77b4',
    'prng': '#d62728',
}
LABELS = {
    'idle': r'Idle $|0\rangle^6$',
    'hadamard': r'Hadamard $|+\rangle^6$',
    'bell': 'Bell pairs',
    'chain': 'Full chain (PRA)',
    'prng': 'PRNG (PCG64)',
}
MARKERS = {'idle': 'v', 'hadamard': '^', 'bell': 's', 'chain': 'o', 'prng': 'D'}


# ============================================================================
# Circuit builders
# ============================================================================

def build_idle_circuit():
    """C1: |0>^6, identity gates, measure."""
    from braket.circuits import Circuit
    c = Circuit()
    for q in range(N_QUBITS):
        c.i(q)
    return c


def build_hadamard_circuit():
    """C2: H on all qubits, measure (no entanglement)."""
    from braket.circuits import Circuit
    c = Circuit()
    for q in range(N_QUBITS):
        c.h(q)
    return c


def build_bell_circuit():
    """C3: 3 independent Bell pairs: (0,1), (2,3), (4,5)."""
    from braket.circuits import Circuit
    c = Circuit()
    for base in [0, 2, 4]:
        c.h(base)
        c.cnot(base, base + 1)
    return c


def build_chain_circuit():
    """C4: H(0) -> CNOT cascade 0-1-2-3-4-5 (same as PRA paper, gamma=0)."""
    from braket.circuits import Circuit
    c = Circuit()
    c.h(0)
    for i in range(N_QUBITS - 1):
        c.cnot(i, i + 1)
    return c


# ============================================================================
# Data collection
# ============================================================================

def run_on_device(device, circuit, shots, label):
    """Run circuit on Braket device, return bitstrings."""
    print(f"    Running {label}: {shots} shots...", end=" ", flush=True)
    t0 = time.time()
    task = device.run(circuit, shots=int(shots))
    result = task.result()
    elapsed = time.time() - t0
    measurements = result.measurements
    bitstrings = [''.join(str(int(b)) for b in row) for row in measurements]
    print(f"done ({elapsed:.1f}s)")
    return bitstrings, elapsed


def generate_sim_bitstrings(circuit_type, n_shots, n_qubits=N_QUBITS):
    """Generate synthetic bitstrings for simulator testing."""
    rng = np.random.default_rng(42 + hash(circuit_type) % 10000)
    bitstrings = []

    if circuit_type == 'idle':
        # ~2% readout error per qubit, slight temporal correlation
        base_error = 0.02
        for i in range(n_shots):
            drift = 0.005 * np.sin(2 * np.pi * i / 500)  # slow drift
            bits = rng.random(n_qubits) < (base_error + drift)
            # Add nearest-neighbor crosstalk
            for q in range(n_qubits - 1):
                if bits[q] and rng.random() < 0.15:
                    bits[q + 1] = True
            bitstrings.append(''.join(str(int(b)) for b in bits))

    elif circuit_type == 'hadamard':
        # ~50% per qubit, slight temporal correlation
        for i in range(n_shots):
            drift = 0.03 * np.sin(2 * np.pi * i / 300)
            probs = 0.5 + drift + rng.normal(0, 0.02, n_qubits)
            # Add readout crosstalk
            bits = rng.random(n_qubits) < probs
            bitstrings.append(''.join(str(int(b)) for b in bits))

    elif circuit_type == 'bell':
        # Correlated within pairs, independent between
        for i in range(n_shots):
            drift = 0.02 * np.sin(2 * np.pi * i / 400)
            bits = np.zeros(n_qubits, dtype=int)
            for base in [0, 2, 4]:
                if rng.random() < 0.5 + drift:
                    bits[base] = 1
                    bits[base + 1] = 1
                # Add some decoherence
                for q in [base, base + 1]:
                    if rng.random() < 0.08:
                        bits[q] = 1 - bits[q]
            bitstrings.append(''.join(str(int(b)) for b in bits))

    elif circuit_type == 'chain':
        # GHZ-like with noise and drift
        for i in range(n_shots):
            drift = 0.04 * np.sin(2 * np.pi * i / 200)
            if rng.random() < 0.45 + drift:
                bits = np.ones(n_qubits, dtype=int)
            else:
                bits = np.zeros(n_qubits, dtype=int)
            # Add per-qubit noise
            for q in range(n_qubits):
                if rng.random() < 0.12:
                    bits[q] = 1 - bits[q]
            bitstrings.append(''.join(str(int(b)) for b in bits))

    return bitstrings


def collect_data(mode='sim'):
    """Collect data from all 4 circuit types."""
    data = {
        'metadata': {
            'experiment': 'idle_qubit_noise_isolation',
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'n_qubits': N_QUBITS,
            'shots_per_circuit': SHOTS,
        },
        'circuits': {}
    }

    circuit_specs = {
        'idle': ('C1: Idle |0>^6', build_idle_circuit),
        'hadamard': ('C2: Hadamard |+>^6', build_hadamard_circuit),
        'bell': ('C3: Bell pairs', build_bell_circuit),
    }

    if mode == 'qpu':
        from braket.aws import AwsDevice
        device = AwsDevice(RIGETTI_ARN)

        total_cost = len(circuit_specs) * (COST_PER_TASK + SHOTS * COST_PER_SHOT)
        print(f"\n  Estimated cost: ${total_cost:.2f} (EUR {total_cost * EUR_PER_USD:.2f})")
        print(f"  Device: Rigetti Ankaa-3")
        print(f"  Circuits: {len(circuit_specs)} new + 1 existing (chain)")

        for key, (label, builder) in circuit_specs.items():
            circuit = builder()
            bitstrings, elapsed = run_on_device(device, circuit, SHOTS, label)
            data['circuits'][key] = {
                'label': label,
                'bitstrings': bitstrings,
                'n_shots': len(bitstrings),
                'elapsed_s': elapsed,
            }

        data['metadata']['cost_usd'] = total_cost
        data['metadata']['cost_eur'] = total_cost * EUR_PER_USD

    else:  # sim mode
        print(f"\n  Simulator mode: generating synthetic bitstrings")
        for key, (label, _) in circuit_specs.items():
            print(f"    Generating {label}: {SHOTS} shots")
            bitstrings = generate_sim_bitstrings(key, SHOTS)
            data['circuits'][key] = {
                'label': label,
                'bitstrings': bitstrings,
                'n_shots': len(bitstrings),
                'elapsed_s': 0.0,
            }

    # C4: Load existing chain data
    if os.path.exists(CHAIN_DATA_FILE):
        print(f"  Loading existing chain data: {CHAIN_DATA_FILE}")
        with open(CHAIN_DATA_FILE) as f:
            chain_data = json.load(f)
        block = chain_data['blocks']['rigetti_boost']
        for m in block['measurements']:
            if abs(m['gamma'] - 0.0) < 0.01:
                data['circuits']['chain'] = {
                    'label': 'C4: Full chain (PRA data)',
                    'bitstrings': m['bitstrings'],
                    'n_shots': len(m['bitstrings']),
                    'elapsed_s': 0.0,
                    'source': CHAIN_DATA_FILE,
                }
                print(f"    Chain: {len(m['bitstrings'])} shots loaded")
                break
    elif mode == 'sim':
        print(f"    Generating chain: {SHOTS} shots (synthetic)")
        bitstrings = generate_sim_bitstrings('chain', SHOTS)
        data['circuits']['chain'] = {
            'label': 'C4: Full chain (synthetic)',
            'bitstrings': bitstrings,
            'n_shots': len(bitstrings),
            'elapsed_s': 0.0,
        }

    # Save raw data
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DATA, 'w') as f:
        json.dump(data, f, indent=1)
    print(f"\n  Data saved: {OUT_DATA}")

    return data


# ============================================================================
# Observable computation
# ============================================================================

def to_bit_array(bitstrings):
    """Convert list of bitstrings to numpy array [n_shots, n_qubits]."""
    return np.array([[int(b) for b in bs] for bs in bitstrings])


def hamming_weights(bitstrings):
    """Hamming weight for each bitstring."""
    return np.array([sum(int(b) for b in bs) for bs in bitstrings])


def qubit_correlations(bits_array):
    """Compute C_ij = <q_i q_j> - <q_i><q_j> for all pairs."""
    n_qubits = bits_array.shape[1]
    means = bits_array.mean(axis=0)
    corr = {}
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            cij = (bits_array[:, i] * bits_array[:, j]).mean() - means[i] * means[j]
            # SE via bootstrap-free formula for binary variables
            n = len(bits_array)
            se = np.sqrt(
                ((bits_array[:, i] * bits_array[:, j] - means[i] * means[j])**2).mean() / n
            )
            corr[(i, j)] = {'c': float(cij), 'se': float(se)}
    return corr


def temporal_autocorrelation(values, max_lag=200):
    """Compute normalized autocorrelation R(tau) for lag = 1..max_lag."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    var = values.var()
    if var < 1e-15:
        return np.arange(1, max_lag + 1), np.zeros(max_lag)

    lags = np.arange(1, min(max_lag + 1, n // 4))
    acf = np.zeros(len(lags))
    for idx, lag in enumerate(lags):
        acf[idx] = np.mean((values[:n-lag] - mean) * (values[lag:] - mean)) / var

    return lags, acf


# ============================================================================
# Variance scaling analysis
# ============================================================================

def variance_scaling(bitstrings, observable='hamming_var'):
    """Compute alpha from inter-batch variance scaling."""
    n = len(bitstrings)
    batch_sizes = [k for k in BATCH_SIZES if k <= n // 3]

    ks, variances = [], []
    for K in batch_sizes:
        nb = n // K
        if observable == 'hamming_var':
            vals = [float(np.var(hamming_weights(bitstrings[b*K:(b+1)*K])))
                    for b in range(nb)]
        elif observable == 'hamming_mean':
            vals = [float(np.mean(hamming_weights(bitstrings[b*K:(b+1)*K])))
                    for b in range(nb)]
        elif observable == 'flip_rate':
            vals = [float(np.mean(hamming_weights(bitstrings[b*K:(b+1)*K])) / N_QUBITS)
                    for b in range(nb)]
        else:
            raise ValueError(f"Unknown observable: {observable}")

        if len(vals) >= 3:
            ks.append(K)
            variances.append(np.var(vals))

    ks = np.array(ks, dtype=float)
    variances = np.array(variances)
    valid = variances > 0

    if valid.sum() < 3:
        return None

    log_k = np.log(ks[valid])
    log_v = np.log(variances[valid] + 1e-30)

    slope, intercept, r, p_reg, se = stats.linregress(log_k, log_v)

    # t-test for H0: alpha = -1
    t_stat = (slope - (-1.0)) / se
    df = len(log_k) - 2
    p_val = 2 * stats.t.sf(abs(t_stat), df)

    return {
        'log_k': log_k, 'log_v': log_v,
        'alpha': slope, 'se': se, 'intercept': intercept,
        'r_squared': r**2, 'p_iid': p_val,
        'n_points': len(log_k),
    }


def shuffled_shots_control(bitstrings, observable='hamming_var', n_perms=N_SHUFFLE):
    """Permutation test: decompose alpha into temporal and distributional."""
    rng = np.random.default_rng(42)
    orig = variance_scaling(bitstrings, observable)
    if orig is None:
        return None

    alpha_perms = []
    for _ in range(n_perms):
        perm = rng.permutation(len(bitstrings))
        bs_perm = [bitstrings[i] for i in perm]
        res = variance_scaling(bs_perm, observable)
        if res is not None:
            alpha_perms.append(res['alpha'])

    alpha_perms = np.array(alpha_perms)
    alpha_shuf_mean = float(np.mean(alpha_perms))
    alpha_shuf_std = float(np.std(alpha_perms))
    ci_low = float(np.percentile(alpha_perms, 2.5))
    ci_high = float(np.percentile(alpha_perms, 97.5))

    delta_T = orig['alpha'] - alpha_shuf_mean  # temporal component
    delta_D = alpha_shuf_mean - (-1.0)          # distributional component

    # p-value: fraction of shuffles with |dev from -1| >= |orig dev|
    orig_dev = abs(orig['alpha'] - (-1.0))
    perm_devs = np.abs(alpha_perms - (-1.0))
    p_shuffle = float(np.mean(perm_devs >= orig_dev))

    return {
        'alpha_orig': orig['alpha'],
        'alpha_orig_se': orig['se'],
        'alpha_shuffled': alpha_shuf_mean,
        'alpha_shuffled_std': alpha_shuf_std,
        'ci_95': [ci_low, ci_high],
        'delta_T': delta_T,
        'delta_D': delta_D,
        'p_shuffle': p_shuffle,
    }


# ============================================================================
# PRNG baseline
# ============================================================================

def generate_prng_baseline(n_shots, n_qubits=N_QUBITS, seed=42):
    """Generate perfect i.i.d. bitstrings as baseline."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_shots, n_qubits))
    return [''.join(str(b) for b in row) for row in bits]


# ============================================================================
# Complete analysis
# ============================================================================

def analyze(data):
    """Run complete analysis pipeline."""
    print("\n" + "=" * 70)
    print("  ANALYSIS: IDLE QUBIT NOISE ISOLATION")
    print("=" * 70)

    results = {}

    # Add PRNG baseline
    circuits = dict(data['circuits'])
    prng_bs = generate_prng_baseline(SHOTS)
    circuits['prng'] = {'bitstrings': prng_bs, 'n_shots': SHOTS, 'label': 'PRNG (PCG64)'}

    circuit_order = ['idle', 'hadamard', 'bell', 'chain', 'prng']
    circuit_order = [c for c in circuit_order if c in circuits]

    # ------------------------------------------------------------------
    # 1. Basic statistics
    # ------------------------------------------------------------------
    print(f"\n{'Circuit':<18} {'N':>6} {'<H>':>6} {'Var(H)':>8} {'Flip%':>7} "
          f"{'Unique':>7} {'000000%':>8}")
    print("-" * 70)

    for key in circuit_order:
        bs = circuits[key]['bitstrings']
        hw = hamming_weights(bs)
        n_unique = len(set(bs))
        zero_frac = bs.count('0' * N_QUBITS) / len(bs)
        label = key.capitalize() if key != 'prng' else 'PRNG'
        print(f"  {label:<16} {len(bs):>6} {hw.mean():>6.3f} {hw.var():>8.4f} "
              f"{hw.mean()/N_QUBITS*100:>6.1f}% {n_unique:>7} {zero_frac*100:>7.1f}%")

    # ------------------------------------------------------------------
    # 2. Variance scaling (alpha)
    # ------------------------------------------------------------------
    print(f"\n{'--- Variance Scaling ---':^70}")
    observables = ['hamming_var', 'hamming_mean']

    for obs in observables:
        print(f"\n  Observable: {obs}")
        print(f"  {'Circuit':<16} {'alpha':>8} {'SE':>6} {'p(iid)':>10} {'Sig':>5}")
        print(f"  {'-'*50}")

        for key in circuit_order:
            bs = circuits[key]['bitstrings']
            res = variance_scaling(bs, obs)
            if res is None:
                print(f"  {key:<16} {'N/A':>8} (insufficient variance)")
                continue
            sig = "***" if res['p_iid'] < 0.001 else "**" if res['p_iid'] < 0.01 else "*" if res['p_iid'] < 0.05 else "ns"
            print(f"  {key:<16} {res['alpha']:>+8.3f} {res['se']:>6.3f} {res['p_iid']:>10.2e} {sig:>5}")

            if key not in results:
                results[key] = {}
            results[key][f'alpha_{obs}'] = res

    # ------------------------------------------------------------------
    # 3. Qubit-pair correlations
    # ------------------------------------------------------------------
    print(f"\n{'--- Qubit-Pair Correlations ---':^70}")

    for key in circuit_order:
        if key == 'prng':
            continue
        bs = circuits[key]['bitstrings']
        bits = to_bit_array(bs)
        corr = qubit_correlations(bits)

        # Summary
        c_vals = [corr[p]['c'] for p in QUBIT_PAIRS]
        c_abs = [abs(c) for c in c_vals]
        n_sig = sum(1 for p in QUBIT_PAIRS if abs(corr[p]['c']) > 2 * corr[p]['se'])

        print(f"\n  {key}: mean|C_ij| = {np.mean(c_abs):.4f}, "
              f"max|C_ij| = {np.max(c_abs):.4f}, "
              f"significant pairs: {n_sig}/{len(QUBIT_PAIRS)}")

        # Adjacent vs non-adjacent
        adj_c = [abs(corr[p]['c']) for p in QUBIT_PAIRS if PAIR_DISTANCES[p] == 1]
        non_adj_c = [abs(corr[p]['c']) for p in QUBIT_PAIRS if PAIR_DISTANCES[p] > 1]
        if adj_c and non_adj_c:
            t_dist, p_dist = stats.mannwhitneyu(adj_c, non_adj_c, alternative='greater')
            print(f"    Adjacent |C|: {np.mean(adj_c):.4f} vs Non-adj: {np.mean(non_adj_c):.4f} "
                  f"(Mann-Whitney p = {p_dist:.3f})")

        if key not in results:
            results[key] = {}
        results[key]['correlations'] = corr

    # ------------------------------------------------------------------
    # 4. Temporal autocorrelation
    # ------------------------------------------------------------------
    print(f"\n{'--- Temporal Autocorrelation ---':^70}")

    for key in circuit_order:
        if key == 'prng':
            continue
        bs = circuits[key]['bitstrings']
        hw = hamming_weights(bs)
        lags, acf = temporal_autocorrelation(hw, max_lag=100)

        # Ljung-Box test (first 10 lags)
        n = len(hw)
        lb_lags = min(10, len(acf))
        lb_stat = n * (n + 2) * sum(acf[k]**2 / (n - k - 1) for k in range(lb_lags))
        lb_p = 1 - stats.chi2.cdf(lb_stat, lb_lags)

        # Correlation time (first lag where |R| < 1/e)
        tau_c = len(acf)  # default: longer than measured
        for idx, r in enumerate(acf):
            if abs(r) < 1.0 / np.e:
                tau_c = lags[idx]
                break

        print(f"  {key}: R(1) = {acf[0]:.4f}, tau_c = {tau_c}, "
              f"Ljung-Box p = {lb_p:.4f} {'(temporal!)' if lb_p < 0.05 else '(i.i.d.)'}")

        if key not in results:
            results[key] = {}
        results[key]['autocorrelation'] = {
            'lags': lags.tolist(), 'acf': acf.tolist(),
            'tau_c': int(tau_c), 'ljung_box_p': float(lb_p),
            'r1': float(acf[0]),
        }

    # ------------------------------------------------------------------
    # 5. Shuffled-shots control
    # ------------------------------------------------------------------
    print(f"\n{'--- Shuffled-Shots Control (hamming_var) ---':^70}")
    print(f"  {'Circuit':<14} {'a_orig':>8} {'a_shuf':>12} {'dT':>8} {'dD':>8} {'Interpretation'}")
    print(f"  {'-'*65}")

    for key in circuit_order:
        if key == 'prng':
            continue
        bs = circuits[key]['bitstrings']
        shuf = shuffled_shots_control(bs, 'hamming_var', N_SHUFFLE)
        if shuf is None:
            print(f"  {key:<14} N/A (insufficient variance)")
            continue

        # Interpretation
        if abs(shuf['delta_D']) < 2 * shuf['alpha_shuffled_std']:
            interp = "Temporal only"
        elif abs(shuf['delta_T']) < 2 * shuf['alpha_orig_se']:
            interp = "Distributional only"
        else:
            interp = "Temporal + distributional"

        print(f"  {key:<14} {shuf['alpha_orig']:>+8.3f} "
              f"{shuf['alpha_shuffled']:>+6.3f}+/-{shuf['alpha_shuffled_std']:.3f} "
              f"{shuf['delta_T']:>+8.3f} {shuf['delta_D']:>+8.3f} {interp}")

        if key not in results:
            results[key] = {}
        results[key]['shuffle_hamming_var'] = shuf

    # ------------------------------------------------------------------
    # 6. Statistical comparisons between circuits
    # ------------------------------------------------------------------
    print(f"\n{'--- Circuit Comparisons (alpha for hamming_var) ---':^70}")

    keys_with_alpha = [k for k in circuit_order
                       if k in results and f'alpha_hamming_var' in results[k]
                       and results[k]['alpha_hamming_var'] is not None]

    for i, k1 in enumerate(keys_with_alpha):
        for k2 in keys_with_alpha[i+1:]:
            a1 = results[k1]['alpha_hamming_var']
            a2 = results[k2]['alpha_hamming_var']
            # Welch's t-test
            t = (a1['alpha'] - a2['alpha']) / np.sqrt(a1['se']**2 + a2['se']**2)
            df = (a1['se']**2 + a2['se']**2)**2 / (
                a1['se']**4 / (a1['n_points'] - 2) + a2['se']**4 / (a2['n_points'] - 2))
            p = 2 * stats.t.sf(abs(t), df)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {k1} vs {k2}: Delta_alpha = {a1['alpha'] - a2['alpha']:+.3f}, "
                  f"t = {t:.2f}, p = {p:.4f} {sig}")

    return results, circuits, circuit_order


# ============================================================================
# Figures
# ============================================================================

def make_figures(results, circuits, circuit_order):
    """Generate all 6 analysis figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    physical_circuits = [c for c in circuit_order if c != 'prng']

    # ==================================================================
    # Figure 1: Correlation matrices (2x2 grid)
    # ==================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Qubit-Pair Correlations $C_{ij}$ by Circuit Type', fontsize=14, fontweight='bold')

    for idx, key in enumerate(physical_circuits[:4]):
        ax = axes[idx // 2, idx % 2]
        if key not in results or 'correlations' not in results[key]:
            ax.set_visible(False)
            continue

        corr = results[key]['correlations']
        mat = np.zeros((N_QUBITS, N_QUBITS))
        for (i, j), vals in corr.items():
            mat[i, j] = vals['c']
            mat[j, i] = vals['c']

        vmax = max(0.01, np.max(np.abs(mat)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(mat, cmap='RdBu_r', norm=norm, aspect='equal')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate
        for i in range(N_QUBITS):
            for j in range(N_QUBITS):
                if i != j:
                    ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center', fontsize=7)

        ax.set_xticks(range(N_QUBITS))
        ax.set_yticks(range(N_QUBITS))
        ax.set_xlabel('Qubit', fontsize=10)
        ax.set_ylabel('Qubit', fontsize=10)
        ax.set_title(LABELS.get(key, key), fontsize=12)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'idle_fig1_correlations.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: idle_fig1_correlations.png")

    # ==================================================================
    # Figure 2: Variance scaling log-log (all circuits overlaid)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Variance Scaling: $\\ln V(K)$ vs $\\ln K$', fontsize=14, fontweight='bold')

    for ax_idx, obs in enumerate(['hamming_var', 'hamming_mean']):
        ax = axes[ax_idx]
        obs_label = 'Var($H$)' if obs == 'hamming_var' else r'$\langle H \rangle$'

        for key in circuit_order:
            rkey = f'alpha_{obs}'
            if key not in results or rkey not in results[key] or results[key][rkey] is None:
                continue
            r = results[key][rkey]
            color = COLORS.get(key, 'black')
            marker = MARKERS.get(key, 'o')
            label_str = f"{LABELS.get(key, key)}: $\\alpha = {r['alpha']:+.3f}$"

            ax.plot(r['log_k'], r['log_v'], marker, color=color, ms=6, alpha=0.7, label=label_str)
            fit_x = np.linspace(r['log_k'].min(), r['log_k'].max(), 50)
            ax.plot(fit_x, r['alpha'] * fit_x + r['intercept'], '-', color=color, lw=1.5, alpha=0.4)

        # i.i.d. reference
        all_log_k = []
        all_log_v = []
        for key in circuit_order:
            rkey = f'alpha_{obs}'
            if key in results and rkey in results[key] and results[key][rkey] is not None:
                all_log_k.extend(results[key][rkey]['log_k'].tolist())
                all_log_v.extend(results[key][rkey]['log_v'].tolist())
        if all_log_k:
            x_ref = np.array([min(all_log_k), max(all_log_k)])
            y_mid = np.mean(all_log_v)
            x_mid = np.mean(all_log_k)
            iid_int = y_mid - (-1.0) * x_mid
            ax.plot(x_ref, -1.0 * x_ref + iid_int, 'k--', lw=2, alpha=0.3,
                    label=r'i.i.d.: $\alpha = -1.0$')

        ax.set_xlabel(r'$\ln(K)$', fontsize=12)
        ax.set_ylabel(r'$\ln(\mathrm{Var})$', fontsize=12)
        ax.set_title(f'Observable: {obs_label}', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'idle_fig2_variance_scaling.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: idle_fig2_variance_scaling.png")

    # ==================================================================
    # Figure 3: Alpha summary bar chart
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(r'Scaling Exponent $\alpha$ Summary', fontsize=14, fontweight='bold')

    for ax_idx, obs in enumerate(['hamming_var', 'hamming_mean']):
        ax = axes[ax_idx]
        obs_label = 'Var($H$)' if obs == 'hamming_var' else r'$\langle H \rangle$'

        names, alphas, ses, colors_list = [], [], [], []
        for key in circuit_order:
            rkey = f'alpha_{obs}'
            if key in results and rkey in results[key] and results[key][rkey] is not None:
                r = results[key][rkey]
                names.append(LABELS.get(key, key))
                alphas.append(r['alpha'])
                ses.append(r['se'])
                colors_list.append(COLORS.get(key, 'gray'))

        x_pos = np.arange(len(names))
        ax.bar(x_pos, alphas, yerr=ses, capsize=5, color=colors_list, alpha=0.7,
               edgecolor='black', lw=0.8)
        ax.axhline(-1.0, color='red', ls='--', lw=2, label=r'i.i.d.: $\alpha = -1.0$')
        ax.axhline(0.0, color='black', ls='-', lw=0.5, alpha=0.3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=8, rotation=15, ha='right')
        ax.set_ylabel(r'$\alpha$', fontsize=12)
        ax.set_title(f'Observable: {obs_label}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'idle_fig3_alpha_summary.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: idle_fig3_alpha_summary.png")

    # ==================================================================
    # Figure 4: Temporal autocorrelation R(tau)
    # ==================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_title('Temporal Autocorrelation $R(\\tau)$ of Hamming Weight', fontsize=13, fontweight='bold')

    for key in physical_circuits:
        if key not in results or 'autocorrelation' not in results[key]:
            continue
        ac = results[key]['autocorrelation']
        lags = np.array(ac['lags'])
        acf = np.array(ac['acf'])
        color = COLORS.get(key, 'black')
        ax.plot(lags[:50], acf[:50], '-', color=color, lw=1.5, alpha=0.8,
                label=f"{LABELS.get(key, key)} ($\\tau_c = {ac['tau_c']}$)")

    # Significance bounds (95% for white noise)
    n_shots = SHOTS
    bound = 1.96 / np.sqrt(n_shots)
    ax.axhline(bound, color='gray', ls=':', alpha=0.5)
    ax.axhline(-bound, color='gray', ls=':', alpha=0.5)
    ax.axhline(0, color='black', ls='-', lw=0.5, alpha=0.3)

    ax.set_xlabel(r'Lag $\tau$ (shots)', fontsize=12)
    ax.set_ylabel(r'$R(\tau)$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'idle_fig4_autocorrelation.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: idle_fig4_autocorrelation.png")

    # ==================================================================
    # Figure 5: Qubit distance vs correlation strength
    # ==================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_title('Correlation Strength vs Qubit Distance', fontsize=13, fontweight='bold')

    for key in physical_circuits:
        if key not in results or 'correlations' not in results[key]:
            continue
        corr = results[key]['correlations']
        dists = [PAIR_DISTANCES[p] for p in QUBIT_PAIRS]
        c_abs = [abs(corr[p]['c']) for p in QUBIT_PAIRS]
        color = COLORS.get(key, 'black')
        marker = MARKERS.get(key, 'o')
        # Jitter x for visibility
        jitter = (list(circuit_order).index(key) - 1.5) * 0.08
        ax.scatter([d + jitter for d in dists], c_abs, c=color, marker=marker,
                   s=50, alpha=0.7, label=LABELS.get(key, key), zorder=3)

    ax.set_xlabel('Qubit distance on chip', fontsize=12)
    ax.set_ylabel(r'$|C_{ij}|$', fontsize=12)
    ax.set_xticks(range(1, N_QUBITS))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'idle_fig5_distance_correlation.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: idle_fig5_distance_correlation.png")

    # ==================================================================
    # Figure 6: Shuffled-shots decomposition
    # ==================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title(r'Shuffled-Shots Decomposition of $\alpha$ (Var($H$))', fontsize=13, fontweight='bold')

    bar_data = []
    for key in physical_circuits:
        if key not in results or 'shuffle_hamming_var' not in results[key]:
            continue
        s = results[key]['shuffle_hamming_var']
        bar_data.append((key, s))

    if bar_data:
        x = np.arange(len(bar_data))
        width = 0.35

        orig_vals = [s['alpha_orig'] for _, s in bar_data]
        shuf_vals = [s['alpha_shuffled'] for _, s in bar_data]
        shuf_errs = [s['alpha_shuffled_std'] for _, s in bar_data]
        bar_colors = [COLORS.get(k, 'gray') for k, _ in bar_data]
        bar_labels = [LABELS.get(k, k) for k, _ in bar_data]

        ax.bar(x - width/2, orig_vals, width, color=bar_colors, alpha=0.9,
               edgecolor='black', lw=0.8, label=r'$\alpha_{\mathrm{orig}}$')
        ax.bar(x + width/2, shuf_vals, width, yerr=shuf_errs, capsize=4,
               color=bar_colors, alpha=0.4, edgecolor='black', lw=0.8, hatch='//',
               label=r'$\alpha_{\mathrm{shuffled}}$')

        ax.axhline(-1.0, color='red', ls='--', lw=2, label=r'i.i.d.: $\alpha = -1.0$')
        ax.axhline(0.0, color='black', ls='-', lw=0.5, alpha=0.3)

        # Annotate delta_T
        for i, (key, s) in enumerate(bar_data):
            ax.annotate(f"$\\Delta\\alpha_T = {s['delta_T']:+.2f}$",
                        xy=(i, max(s['alpha_orig'], s['alpha_shuffled']) + 0.05),
                        ha='center', fontsize=8, fontstyle='italic')

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel(r'$\alpha$', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'idle_fig6_shuffle_decomposition.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: idle_fig6_shuffle_decomposition.png")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Idle Qubit Noise Isolation Experiment')
    parser.add_argument('--mode', choices=['sim', 'qpu'], default='sim',
                        help='Run mode: sim (free) or qpu (Rigetti, ~$11)')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Analyze existing JSON data file')
    args = parser.parse_args()

    print("=" * 70)
    print("  IDLE QUBIT NOISE ISOLATION EXPERIMENT")
    print("=" * 70)

    if args.analyze:
        print(f"  Loading data: {args.analyze}")
        with open(args.analyze) as f:
            data = json.load(f)
    else:
        data = collect_data(args.mode)

    results, circuits, circuit_order = analyze(data)

    print(f"\n{'--- Generating Figures ---':^70}")
    make_figures(results, circuits, circuit_order)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
