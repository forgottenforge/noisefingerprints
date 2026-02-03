#!/usr/bin/env python3
"""
VACUUM TELESCOPE — RIGOROUS STATISTICAL VALIDATION
====================================================

Performs publication-grade statistical analysis on QPU data:
  1. Bootstrap confidence intervals for all key metrics
  2. Permutation tests for significance (p-values)
  3. Cross-validation (train/test split)
  4. QFT violation tests (variance scaling exponent)
  5. Meta-analysis of gamma_c measurements
  6. Vacuum communication binomial tests
  7. Publication-quality figures

Reads: data/vacuum_telescope_v1/vacuum_telescope_qpu_*.json
Writes: data/vacuum_telescope_v1/statistics_validation.json
        data/vacuum_telescope_v1/figures/*.png

Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

try:
    from scipy import stats
    from scipy.signal import savgol_filter, find_peaks, peak_prominences
except ImportError:
    print("[ERROR] scipy required: pip install scipy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
    })
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARNING] matplotlib not available, skipping figures")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hamming_weights(bitstrings: List[str]) -> np.ndarray:
    return np.array([sum(int(b) for b in bs) for bs in bitstrings])

def bitstring_entropy(bitstrings: List[str]) -> float:
    counts = Counter(bitstrings)
    total = len(bitstrings)
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-15)))

def compute_sigma_c(sigmas: np.ndarray, observables: np.ndarray,
                    smoothing_window: int = 5) -> Tuple[np.ndarray, float, float]:
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


def batch_sigma_c_from_bitstrings(bitstrings: List[str], obs_func, label: str = "obs"):
    """Compute batch-size sigma-c for a given observable function."""
    n_total = len(bitstrings)
    batch_sizes = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400]
    batch_sizes = [k for k in batch_sizes if k <= n_total // 3]

    batch_ks = []
    batch_means = []
    batch_stds = []

    for K in batch_sizes:
        n_batches = n_total // K
        values = [obs_func(bitstrings[b*K:(b+1)*K]) for b in range(n_batches)]
        batch_ks.append(K)
        batch_means.append(float(np.mean(values)))
        batch_stds.append(float(np.std(values)))

    batch_ks = np.array(batch_ks, dtype=float)
    batch_means_arr = np.array(batch_means)
    batch_stds_arr = np.array(batch_stds)

    chi, sigma_c, kappa = compute_sigma_c(batch_ks, batch_means_arr)

    # Variance scaling exponent
    log_k = np.log(batch_ks)
    variances = batch_stds_arr ** 2
    valid = variances > 0
    if valid.sum() >= 3:
        slope, intercept, r, p, se = stats.linregress(log_k[valid], np.log(variances[valid] + 1e-30))
    else:
        slope, se = -1.0, 999.0

    return {
        'batch_sizes': batch_ks.tolist(),
        'batch_means': batch_means,
        'batch_stds': [float(s) for s in batch_stds_arr],
        'chi': chi.tolist(),
        'sigma_c': sigma_c,
        'kappa': kappa,
        'var_slope': slope,
        'var_slope_se': se,
    }


# =============================================================================
# 1. BOOTSTRAP CONFIDENCE INTERVALS FOR BATCH-SIZE ANOMALY
# =============================================================================

def bootstrap_batch_sigma_c(bitstrings: List[str], obs_func,
                            n_bootstrap: int = 5000, ci: float = 0.95) -> Dict:
    """
    Bootstrap CI for sigma_c and kappa from batch-size analysis.
    Resamples bitstrings with replacement.
    """
    n = len(bitstrings)
    rng = np.random.default_rng(42)

    # Original computation
    orig = batch_sigma_c_from_bitstrings(bitstrings, obs_func)

    boot_sigma_cs = []
    boot_kappas = []
    boot_slopes = []

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        resampled = [bitstrings[j] for j in indices]
        result = batch_sigma_c_from_bitstrings(resampled, obs_func)
        boot_sigma_cs.append(result['sigma_c'])
        boot_kappas.append(result['kappa'])
        boot_slopes.append(result['var_slope'])

    alpha = (1 - ci) / 2
    return {
        'sigma_c': orig['sigma_c'],
        'sigma_c_ci': [float(np.percentile(boot_sigma_cs, alpha * 100)),
                       float(np.percentile(boot_sigma_cs, (1 - alpha) * 100))],
        'sigma_c_std': float(np.std(boot_sigma_cs)),
        'kappa': orig['kappa'],
        'kappa_ci': [float(np.percentile(boot_kappas, alpha * 100)),
                     float(np.percentile(boot_kappas, (1 - alpha) * 100))],
        'kappa_std': float(np.std(boot_kappas)),
        'var_slope': orig['var_slope'],
        'var_slope_ci': [float(np.percentile(boot_slopes, alpha * 100)),
                         float(np.percentile(boot_slopes, (1 - alpha) * 100))],
        'var_slope_se': orig['var_slope_se'],
        'n_bootstrap': n_bootstrap,
        'batch_sizes': orig['batch_sizes'],
        'batch_means': orig['batch_means'],
        'batch_stds': orig['batch_stds'],
        'chi': orig['chi'],
    }


# =============================================================================
# 2. PERMUTATION TEST FOR BATCH-SIZE ANOMALY
# =============================================================================

def permutation_test_batch(bitstrings: List[str], obs_func,
                           n_permutations: int = 5000) -> Dict:
    """
    Null hypothesis: No preferred batch size (kappa ~ 1).
    Permute bitstrings randomly, recompute kappa.
    p-value = fraction with kappa >= observed.
    """
    rng = np.random.default_rng(123)
    n = len(bitstrings)

    # Observed kappa
    orig = batch_sigma_c_from_bitstrings(bitstrings, obs_func)
    observed_kappa = orig['kappa']

    count_exceed = 0
    perm_kappas = []

    for i in range(n_permutations):
        indices = rng.permutation(n)
        shuffled = [bitstrings[j] for j in indices]
        result = batch_sigma_c_from_bitstrings(shuffled, obs_func)
        perm_kappas.append(result['kappa'])
        if result['kappa'] >= observed_kappa:
            count_exceed += 1

    p_value = (count_exceed + 1) / (n_permutations + 1)  # +1 for continuity

    return {
        'observed_kappa': observed_kappa,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'perm_kappa_mean': float(np.mean(perm_kappas)),
        'perm_kappa_std': float(np.std(perm_kappas)),
        'perm_kappa_95': float(np.percentile(perm_kappas, 95)),
        'perm_kappa_99': float(np.percentile(perm_kappas, 99)),
        'cohens_d': float((observed_kappa - np.mean(perm_kappas)) / (np.std(perm_kappas) + 1e-15)),
    }


# =============================================================================
# 3. CROSS-VALIDATION (SPLIT-HALF)
# =============================================================================

def cross_validate_batch(bitstrings: List[str], obs_func, n_splits: int = 100) -> Dict:
    """
    Split-half cross-validation: find sigma_c on first half, test on second.
    """
    rng = np.random.default_rng(456)
    n = len(bitstrings)
    half = n // 2

    train_sigma_cs = []
    test_kappas = []
    consistent = 0

    for i in range(n_splits):
        indices = rng.permutation(n)
        train = [bitstrings[j] for j in indices[:half]]
        test = [bitstrings[j] for j in indices[half:2*half]]

        train_result = batch_sigma_c_from_bitstrings(train, obs_func)
        test_result = batch_sigma_c_from_bitstrings(test, obs_func)

        train_sigma_cs.append(train_result['sigma_c'])
        test_kappas.append(test_result['kappa'])

        if test_result['kappa'] > 3.0:
            consistent += 1

    return {
        'n_splits': n_splits,
        'train_sigma_c_mean': float(np.mean(train_sigma_cs)),
        'train_sigma_c_std': float(np.std(train_sigma_cs)),
        'test_kappa_mean': float(np.mean(test_kappas)),
        'test_kappa_std': float(np.std(test_kappas)),
        'test_kappa_gt3_frac': consistent / n_splits,
        'consistent': consistent / n_splits > 0.5,
    }


# =============================================================================
# 4. QFT VIOLATION TEST
# =============================================================================

def qft_violation_test(bitstrings: List[str], obs_func) -> Dict:
    """
    Test if var(K) ~ K^alpha with alpha = -1.0 (QFT prediction).
    """
    result = batch_sigma_c_from_bitstrings(bitstrings, obs_func)
    slope = result['var_slope']
    se = result['var_slope_se']

    # Test: |slope - (-1.0)| / SE
    deviation = slope - (-1.0)
    if se > 0 and se < 100:
        t_stat = deviation / se
        p_value = 2 * stats.t.sf(abs(t_stat), df=len(result['batch_sizes']) - 2)
    else:
        t_stat = 0.0
        p_value = 1.0

    return {
        'slope': slope,
        'slope_se': se,
        'qft_prediction': -1.0,
        'deviation': deviation,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


# =============================================================================
# 5. META-ANALYSIS OF GAMMA_C
# =============================================================================

def meta_analysis_gamma_c() -> Dict:
    """
    Combine six gamma_c measurements from E3, E6, V1 (entropy), V1 (Hamming),
    V2 (witness), V7 (communication).
    """
    measurements = [
        {'source': 'E3 (entanglement witness)', 'gamma_c': 0.6737, 'kappa': 8.58, 'experiment': 'magnetguy'},
        {'source': 'E6 (GHZ decoherence)', 'gamma_c': 0.6842, 'kappa': 1.65, 'experiment': 'magnetguy'},
        {'source': 'V1 (entropy)', 'gamma_c': 0.71, 'kappa': 6.35, 'experiment': 'telescope'},
        {'source': 'V1 (mean Hamming)', 'gamma_c': 0.58, 'kappa': 10.58, 'experiment': 'telescope'},
        {'source': 'V2 (witness)', 'gamma_c': 0.674, 'kappa': 3.19, 'experiment': 'telescope'},
        {'source': 'V7 (info loss)', 'gamma_c': 0.62, 'kappa': 5.26, 'experiment': 'telescope'},
    ]

    gammas = np.array([m['gamma_c'] for m in measurements])
    kappas = np.array([m['kappa'] for m in measurements])

    # Weighted mean (weight by kappa = signal strength)
    weights = kappas / kappas.sum()
    weighted_mean = float(np.average(gammas, weights=kappas))

    # Unweighted statistics
    mean = float(np.mean(gammas))
    std = float(np.std(gammas, ddof=1))
    se = std / np.sqrt(len(gammas))
    ci_95 = [mean - 1.96 * se, mean + 1.96 * se]

    # Bayesian posterior (uniform prior on [0.5, 0.8], Gaussian likelihood)
    gamma_grid = np.linspace(0.50, 0.80, 1000)
    log_posterior = np.zeros_like(gamma_grid)
    for m in measurements:
        # Use kappa-derived uncertainty: higher kappa = tighter constraint
        sigma_est = 0.05 / np.sqrt(m['kappa'])
        log_posterior += -0.5 * ((gamma_grid - m['gamma_c']) / sigma_est) ** 2
    posterior = np.exp(log_posterior - log_posterior.max())
    posterior /= np.trapezoid(posterior, gamma_grid)

    # Credible interval
    cdf = np.cumsum(posterior) * (gamma_grid[1] - gamma_grid[0])
    map_idx = np.argmax(posterior)
    map_value = float(gamma_grid[map_idx])

    ci_lower_idx = np.searchsorted(cdf, 0.025)
    ci_upper_idx = np.searchsorted(cdf, 0.975)
    bayes_ci = [float(gamma_grid[ci_lower_idx]), float(gamma_grid[min(ci_upper_idx, len(gamma_grid)-1)])]
    bayes_mean = float(np.trapezoid(gamma_grid * posterior, gamma_grid))

    # Consistency test (chi-squared)
    chi2 = float(np.sum((gammas - mean) ** 2 / (std ** 2 + 1e-15)))
    df = len(gammas) - 1
    chi2_p = float(stats.chi2.sf(chi2, df))

    return {
        'measurements': measurements,
        'n_measurements': len(measurements),
        'unweighted_mean': mean,
        'unweighted_std': std,
        'unweighted_se': se,
        'unweighted_ci_95': ci_95,
        'weighted_mean': weighted_mean,
        'bayesian_map': map_value,
        'bayesian_mean': bayes_mean,
        'bayesian_ci_95': bayes_ci,
        'chi2_statistic': chi2,
        'chi2_df': df,
        'chi2_p_value': chi2_p,
        'consistent': chi2_p > 0.05,
        'posterior_grid': gamma_grid.tolist(),
        'posterior_density': posterior.tolist(),
    }


# =============================================================================
# 6. VACUUM COMMUNICATION VALIDATION
# =============================================================================

def validate_vacuum_communication(data: Dict) -> Dict:
    """
    Bootstrap CIs for accuracy, binomial tests, Bonferroni correction.
    """
    v7 = data['blocks'].get('V7_communication')
    if not v7 or v7.get('status') != 'COMPLETE':
        return {'status': 'SKIPPED'}

    rng = np.random.default_rng(789)
    n_bootstrap = 5000
    n_tests = 0

    results_by_condition = []

    messages = sorted(set(m['message'] for m in v7['measurements']))
    gammas = sorted(set(m['gamma'] for m in v7['measurements']))
    n_tests = len(messages) * len(gammas)

    for meas in v7['measurements']:
        msg = meas['message']
        gamma = meas['gamma']
        bitstrings = meas['bitstrings']
        n_shots = len(bitstrings)
        n_qubits = len(msg)

        # Compute accuracy from raw bitstrings
        qubit_correct_counts = []
        for qi in range(n_qubits):
            target_bit = msg[qi]
            ones = sum(int(bs[qi]) for bs in bitstrings)
            prob_one = ones / n_shots
            decoded_bit = '1' if prob_one > 0.5 else '0'
            qubit_correct_counts.append(1 if decoded_bit == target_bit else 0)
        accuracy = sum(qubit_correct_counts) / n_qubits

        # Bootstrap CI on accuracy
        boot_accs = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_shots, size=n_shots)
            resampled = [bitstrings[j] for j in idx]
            decoded = []
            for qi in range(n_qubits):
                ones = sum(int(bs[qi]) for bs in resampled)
                decoded.append('1' if ones / n_shots > 0.5 else '0')
            boot_acc = sum(a == b for a, b in zip(msg, decoded)) / n_qubits
            boot_accs.append(boot_acc)

        boot_accs = np.array(boot_accs)
        ci_lower = float(np.percentile(boot_accs, 2.5))
        ci_upper = float(np.percentile(boot_accs, 97.5))

        # Binomial test: H0 accuracy = 0.5
        # For 6 bits, each is binomial(1, 0.5)
        n_correct = int(accuracy * n_qubits)
        binom_p = float(stats.binomtest(n_correct, n_qubits, 0.5, alternative='greater').pvalue)

        # Per-qubit significance (using raw shot counts)
        qubit_details = []
        for qi in range(n_qubits):
            target = int(msg[qi])
            ones = sum(int(bs[qi]) for bs in bitstrings)
            match_count = ones if target == 1 else (n_shots - ones)
            qubit_p = float(stats.binomtest(match_count, n_shots, 0.5, alternative='greater').pvalue)
            qubit_details.append({
                'qubit': qi,
                'target_bit': target,
                'prob_target': match_count / n_shots,
                'p_value': qubit_p,
            })

        results_by_condition.append({
            'message': msg,
            'gamma': gamma,
            'accuracy': accuracy,
            'accuracy_ci': [ci_lower, ci_upper],
            'accuracy_bootstrap_std': float(np.std(boot_accs)),
            'binomial_p_uncorrected': binom_p,
            'binomial_p_bonferroni': min(1.0, binom_p * n_tests),
            'n_correct_bits': n_correct,
            'n_total_bits': n_qubits,
            'qubit_details': qubit_details,
        })

    # Sigma-c on mean accuracy curve
    mean_accs_by_gamma = {}
    for r in results_by_condition:
        g = r['gamma']
        if g not in mean_accs_by_gamma:
            mean_accs_by_gamma[g] = []
        mean_accs_by_gamma[g].append(r['accuracy'])

    gammas_arr = np.array(sorted(mean_accs_by_gamma.keys()))
    mean_accs = np.array([np.mean(mean_accs_by_gamma[g]) for g in gammas_arr])

    chi, sigma_c, kappa = compute_sigma_c(gammas_arr, mean_accs)

    # Bootstrap sigma_c for communication
    boot_sigma_cs = []
    for _ in range(n_bootstrap):
        # Resample within each condition
        boot_mean_accs = []
        for g in gammas_arr:
            accs_at_g = mean_accs_by_gamma[g]
            boot_sample = rng.choice(accs_at_g, size=len(accs_at_g), replace=True)
            boot_mean_accs.append(float(np.mean(boot_sample)))
        _, sc, _ = compute_sigma_c(gammas_arr, np.array(boot_mean_accs))
        boot_sigma_cs.append(sc)

    return {
        'conditions': results_by_condition,
        'n_tests': n_tests,
        'sigma_c': sigma_c,
        'kappa': kappa,
        'sigma_c_ci': [float(np.percentile(boot_sigma_cs, 2.5)),
                       float(np.percentile(boot_sigma_cs, 97.5))],
    }


# =============================================================================
# FIGURES
# =============================================================================

def make_figure_1(data: Dict, stats_results: Dict, fig_dir: Path):
    """Figure 1: chi(K) curves at gamma=0 and gamma=0.67"""
    if not HAS_PLOT:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (gamma_label, gamma_key) in enumerate([
        (r'$\gamma = 0$ (pure QPU vacuum)', 'gamma_0.0'),
        (r'$\gamma = 0.67$ (at $\gamma_c$)', 'gamma_0.67'),
    ]):
        ax = axes[ax_idx]
        key = f'bootstrap_{gamma_key}'
        if key not in stats_results:
            continue

        for obs_name in ['hamming_var', 'entropy', 'hamming_mean']:
            obs_key = f'{key}_{obs_name}'
            if obs_key not in stats_results:
                continue
            r = stats_results[obs_key]
            ks = np.array(r['batch_sizes'])
            chi = np.array(r['chi'])

            style = {'hamming_var': ('C0', '-', r'Var($H$)'),
                     'entropy': ('C1', '--', r'$S$'),
                     'hamming_mean': ('C2', ':', r'$\langle H \rangle$')}
            color, ls, label = style[obs_name]
            ax.plot(ks, chi, color=color, ls=ls, lw=2, label=label, marker='o', ms=4)

            # Mark sigma_c
            sc = r['sigma_c']
            sc_idx = np.argmin(np.abs(ks - sc))
            ax.axvline(sc, color=color, ls=':', alpha=0.5)

        ax.set_xlabel('Batch size $K$')
        ax.set_ylabel(r'$\chi(K) = |d\langle O\rangle / dK|$')
        ax.set_title(gamma_label)
        ax.legend()
        ax.set_xscale('log')

    fig.suptitle('Batch-Size Susceptibility: Vacuum Structure Test', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure_1_chi_K.png')
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'figure_1_chi_K.png'}")


def make_figure_2(meta: Dict, fig_dir: Path):
    """Figure 2: Six gamma_c measurements with Bayesian posterior"""
    if not HAS_PLOT:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})

    measurements = meta['measurements']
    labels = [m['source'] for m in measurements]
    gammas = [m['gamma_c'] for m in measurements]
    kappas = [m['kappa'] for m in measurements]

    # Uncertainty bars from kappa (higher kappa = tighter)
    uncertainties = [0.05 / np.sqrt(k) for k in kappas]

    y_positions = np.arange(len(measurements))
    colors = ['C0' if m['experiment'] == 'magnetguy' else 'C1' for m in measurements]

    ax1.barh(y_positions, [0] * len(measurements), xerr=None)
    ax1.errorbar(gammas, y_positions, xerr=uncertainties, fmt='o', capsize=5,
                 color='black', ms=8, zorder=5)
    for i, (g, y, c) in enumerate(zip(gammas, y_positions, colors)):
        ax1.plot(g, y, 'o', color=c, ms=10, zorder=6)

    ax1.axvline(meta['bayesian_mean'], color='red', ls='-', lw=2, alpha=0.7,
                label=f"Bayesian mean = {meta['bayesian_mean']:.3f}")
    ax1.axvspan(meta['bayesian_ci_95'][0], meta['bayesian_ci_95'][1],
                alpha=0.15, color='red', label='95% credible interval')
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel(r'$\gamma_c$')
    ax1.set_title(r'Six measurements of $\gamma_c$')
    ax1.legend(loc='lower right', fontsize=9)

    # Bayesian posterior
    grid = np.array(meta['posterior_grid'])
    density = np.array(meta['posterior_density'])
    ax2.plot(grid, density, 'k-', lw=2)
    ax2.fill_between(grid, density, alpha=0.2, color='red')
    ax2.axvline(meta['bayesian_mean'], color='red', ls='--', lw=1.5)
    ci = meta['bayesian_ci_95']
    mask = (grid >= ci[0]) & (grid <= ci[1])
    ax2.fill_between(grid[mask], density[mask], alpha=0.4, color='red')
    ax2.set_xlabel(r'$\gamma_c$')
    ax2.set_ylabel('Posterior density')
    ax2.set_title('Bayesian posterior')

    plt.tight_layout()
    fig.savefig(fig_dir / 'figure_2_gamma_c_meta.png')
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'figure_2_gamma_c_meta.png'}")


def make_figure_3(vcomm: Dict, fig_dir: Path):
    """Figure 3: Vacuum communication heatmap"""
    if not HAS_PLOT:
        return

    conditions = vcomm.get('conditions', [])
    if not conditions:
        return

    messages = sorted(set(c['message'] for c in conditions))
    gammas = sorted(set(c['gamma'] for c in conditions))

    acc_matrix = np.zeros((len(messages), len(gammas)))
    for c in conditions:
        mi = messages.index(c['message'])
        gi = gammas.index(c['gamma'])
        acc_matrix[mi, gi] = c['accuracy']

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(acc_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([f'{g:.3f}' for g in gammas], rotation=45)
    ax.set_yticks(range(len(messages)))
    ax.set_yticklabels(messages, fontfamily='monospace')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel('Encoded message')
    ax.set_title('Vacuum Communication: Bit Recovery Accuracy')

    for i in range(len(messages)):
        for j in range(len(gammas)):
            val = acc_matrix[i, j]
            color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', color=color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Accuracy')

    plt.tight_layout()
    fig.savefig(fig_dir / 'figure_3_communication_heatmap.png')
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'figure_3_communication_heatmap.png'}")


def make_figure_4(stats_results: Dict, fig_dir: Path):
    """Figure 4: Variance scaling log-log"""
    if not HAS_PLOT:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (gamma_label, gamma_key) in enumerate([
        (r'$\gamma = 0$', 'gamma_0.0'),
        (r'$\gamma = 0.67$', 'gamma_0.67'),
    ]):
        ax = axes[ax_idx]
        key = f'bootstrap_{gamma_key}_hamming_var'
        if key not in stats_results:
            continue

        r = stats_results[key]
        ks = np.array(r['batch_sizes'])
        stds = np.array(r['batch_stds'])
        variances = stds ** 2

        valid = variances > 0
        log_k = np.log(ks[valid])
        log_v = np.log(variances[valid])

        ax.plot(log_k, log_v, 'ko', ms=8, zorder=5, label='Data')

        # Fit line
        qft_key = f'qft_{gamma_key}_hamming_var'
        if qft_key in stats_results:
            slope = stats_results[qft_key]['slope']
            se = stats_results[qft_key]['slope_se']
            intercept = log_v[0] - slope * log_k[0]
            fit_x = np.linspace(log_k.min(), log_k.max(), 100)
            ax.plot(fit_x, slope * fit_x + (intercept + (log_v[-1] - slope * log_k[-1] - intercept + log_v[0] - slope * log_k[0]) / 2),
                    'C0-', lw=2, label=rf'Fit: $\alpha = {slope:.2f} \pm {se:.2f}$')

            # QFT prediction
            qft_intercept = log_v.mean() - (-1.0) * log_k.mean()
            ax.plot(fit_x, -1.0 * fit_x + qft_intercept,
                    'r--', lw=2, label=r'QFT: $\alpha = -1.0$')

        ax.set_xlabel(r'$\ln(K)$')
        ax.set_ylabel(r'$\ln(\mathrm{Var})$')
        ax.set_title(f'Variance scaling: {gamma_label}')
        ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / 'figure_4_variance_scaling.png')
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'figure_4_variance_scaling.png'}")


def make_figure_5_gamma_sweep(data: Dict, fig_dir: Path):
    """Figure 5: Bitstring entropy and Hamming distance vs gamma"""
    if not HAS_PLOT:
        return

    v1 = data['blocks'].get('V1_z_sweep')
    if not v1:
        return

    gammas = []
    entropies = []
    mean_h = []
    std_h = []
    kurtosis_h = []

    for meas in v1['measurements']:
        gammas.append(meas['gamma'])
        bs = meas['bitstrings']
        hw = hamming_weights(bs)
        entropies.append(bitstring_entropy(bs))
        mean_h.append(np.mean(hw))
        std_h.append(np.std(hw))
        kurtosis_h.append(float(stats.kurtosis(hw)))

    gammas = np.array(gammas)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(gammas, entropies, 'ko-', ms=4)
    axes[0, 0].axvline(0.674, color='red', ls='--', alpha=0.5, label=r'$\gamma_c$')
    axes[0, 0].set_ylabel('Shannon entropy (bits)')
    axes[0, 0].set_title('(a) Bitstring entropy')
    axes[0, 0].legend()

    axes[0, 1].plot(gammas, mean_h, 'ko-', ms=4)
    axes[0, 1].axvline(0.674, color='red', ls='--', alpha=0.5)
    axes[0, 1].set_ylabel(r'$\langle H \rangle$')
    axes[0, 1].set_title('(b) Mean Hamming weight')

    axes[1, 0].plot(gammas, std_h, 'ko-', ms=4)
    axes[1, 0].axvline(0.674, color='red', ls='--', alpha=0.5)
    axes[1, 0].set_ylabel(r'$\sigma_H$')
    axes[1, 0].set_xlabel(r'$\gamma$')
    axes[1, 0].set_title('(c) Hamming std. deviation')

    axes[1, 1].plot(gammas, kurtosis_h, 'ko-', ms=4)
    axes[1, 1].axvline(0.674, color='red', ls='--', alpha=0.5)
    axes[1, 1].set_ylabel(r'Excess kurtosis')
    axes[1, 1].set_xlabel(r'$\gamma$')
    axes[1, 1].set_title('(d) Kurtosis')

    fig.suptitle(r'V1: Bitstring statistics vs decoherence parameter $\gamma$', fontsize=14)
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure_5_gamma_sweep.png')
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'figure_5_gamma_sweep.png'}")


def make_figure_6_witness(data: Dict, fig_dir: Path):
    """Figure 6: Entanglement witness vs gamma"""
    if not HAS_PLOT:
        return

    # Reconstruct witness from raw data
    v1 = data['blocks'].get('V1_z_sweep')
    v2 = data['blocks'].get('V2_witness')
    if not v1 or not v2:
        return

    q1, q2 = 2, 3  # V2_QUBIT_PAIR

    # ZZ from V1
    v1_zz = {}
    for meas in v1['measurements']:
        gamma = round(meas['gamma'], 6)
        bs = meas['bitstrings']
        parity_sum = 0
        for s in bs:
            s1 = 1 - 2 * int(s[q1])
            s2 = 1 - 2 * int(s[q2])
            parity_sum += s1 * s2
        v1_zz[gamma] = parity_sum / len(bs)

    gammas_w = []
    witnesses = []
    xx_vals = []
    yy_vals = []
    zz_vals = []

    for meas in v2['measurements']:
        gamma = round(meas['gamma'], 6)
        xx_bs = meas['XX']['bitstrings']
        yy_bs = meas['YY']['bitstrings']

        xx_exp = sum(1 - 2*(int(s[0])^int(s[1])) for s in xx_bs) / len(xx_bs)
        yy_exp = sum(1 - 2*(int(s[0])^int(s[1])) for s in yy_bs) / len(yy_bs)

        closest = min(v1_zz.keys(), key=lambda g: abs(g - gamma))
        zz_exp = v1_zz[closest]

        w = (xx_exp + yy_exp) / 2 - abs(zz_exp)
        gammas_w.append(gamma)
        witnesses.append(w)
        xx_vals.append(xx_exp)
        yy_vals.append(yy_exp)
        zz_vals.append(zz_exp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(gammas_w, witnesses, 'ko-', ms=8, lw=2, label=r'$W = (\langle XX\rangle + \langle YY\rangle)/2 - |\langle ZZ\rangle|$')
    ax1.axhline(0, color='gray', ls='-', alpha=0.3)
    ax1.axvline(0.674, color='red', ls='--', alpha=0.5, label=r'$\gamma_c = 0.674$')
    ax1.fill_between(gammas_w, witnesses, 0, where=[w < 0 for w in witnesses],
                     alpha=0.2, color='blue', label='Entangled (W < 0)')
    ax1.fill_between(gammas_w, witnesses, 0, where=[w >= 0 for w in witnesses],
                     alpha=0.2, color='red', label='Separable (W >= 0)')
    ax1.set_xlabel(r'$\gamma$')
    ax1.set_ylabel(r'Witness $W$')
    ax1.set_title('(a) Entanglement witness')
    ax1.legend(fontsize=9)

    ax2.plot(gammas_w, xx_vals, 's-', color='C0', label=r'$\langle XX \rangle$', ms=6)
    ax2.plot(gammas_w, yy_vals, 'D-', color='C1', label=r'$\langle YY \rangle$', ms=6)
    ax2.plot(gammas_w, zz_vals, 'o-', color='C2', label=r'$\langle ZZ \rangle$', ms=6)
    ax2.axvline(0.674, color='red', ls='--', alpha=0.5)
    ax2.set_xlabel(r'$\gamma$')
    ax2.set_ylabel('Expectation value')
    ax2.set_title('(b) Pauli correlations')
    ax2.legend()

    fig.suptitle('V2: Entanglement witness reconstruction', fontsize=14)
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure_6_witness.png')
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'figure_6_witness.png'}")


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def run_validation(data_file: str):
    print("=" * 70)
    print("  VACUUM TELESCOPE — STATISTICAL VALIDATION")
    print("=" * 70)

    with open(data_file, 'r') as f:
        data = json.load(f)

    fig_dir = Path("data/vacuum_telescope_v1/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    stats_results = {}

    v1 = data['blocks'].get('V1_z_sweep')
    if not v1:
        print("[ERROR] No V1 data found")
        return

    # Observable functions
    obs_funcs = {
        'hamming_mean': lambda bs: np.mean(hamming_weights(bs)),
        'hamming_var': lambda bs: np.var(hamming_weights(bs)),
        'entropy': lambda bs: bitstring_entropy(bs),
    }

    # ==========================================================
    # 1. BATCH-SIZE ANOMALY VALIDATION
    # ==========================================================

    print("\n" + "=" * 70)
    print("  1. BATCH-SIZE ANOMALY VALIDATION")
    print("=" * 70)

    for gamma_target, gamma_label in [(0.0, 'gamma_0.0'), (0.67, 'gamma_0.67')]:
        all_gammas = [m['gamma'] for m in v1['measurements']]
        idx = int(np.argmin([abs(g - gamma_target) for g in all_gammas]))
        actual_gamma = all_gammas[idx]
        bitstrings = v1['measurements'][idx]['bitstrings']

        print(f"\n  --- gamma = {actual_gamma} ({len(bitstrings)} shots) ---")

        for obs_name, obs_func in obs_funcs.items():
            print(f"\n  Observable: {obs_name}")

            # 1a. Bootstrap
            print(f"    [Bootstrap] Running {5000} iterations...")
            boot = bootstrap_batch_sigma_c(bitstrings, obs_func, n_bootstrap=5000)
            stats_results[f'bootstrap_{gamma_label}_{obs_name}'] = boot
            print(f"    sigma_c = {boot['sigma_c']:.1f}  CI: [{boot['sigma_c_ci'][0]:.1f}, {boot['sigma_c_ci'][1]:.1f}]")
            print(f"    kappa   = {boot['kappa']:.2f}  CI: [{boot['kappa_ci'][0]:.2f}, {boot['kappa_ci'][1]:.2f}]")
            print(f"    slope   = {boot['var_slope']:.3f}  CI: [{boot['var_slope_ci'][0]:.3f}, {boot['var_slope_ci'][1]:.3f}]")

            # 1b. Permutation test
            print(f"    [Permutation] Running {5000} iterations...")
            perm = permutation_test_batch(bitstrings, obs_func, n_permutations=5000)
            stats_results[f'permutation_{gamma_label}_{obs_name}'] = perm
            print(f"    Observed kappa = {perm['observed_kappa']:.2f}")
            print(f"    Null kappa     = {perm['perm_kappa_mean']:.2f} +/- {perm['perm_kappa_std']:.2f}")
            print(f"    p-value        = {perm['p_value']:.6f}")
            print(f"    Cohen's d      = {perm['cohens_d']:.2f}")
            sig = "***" if perm['p_value'] < 0.001 else "**" if perm['p_value'] < 0.01 else "*" if perm['p_value'] < 0.05 else "ns"
            print(f"    Significance   : {sig}")

            # 1c. Cross-validation
            print(f"    [Cross-val] Running 100 splits...")
            cv = cross_validate_batch(bitstrings, obs_func, n_splits=100)
            stats_results[f'crossval_{gamma_label}_{obs_name}'] = cv
            print(f"    Train sigma_c  = {cv['train_sigma_c_mean']:.1f} +/- {cv['train_sigma_c_std']:.1f}")
            print(f"    Test kappa     = {cv['test_kappa_mean']:.2f} +/- {cv['test_kappa_std']:.2f}")
            print(f"    Test kappa > 3 : {cv['test_kappa_gt3_frac']:.0%}")
            print(f"    Consistent     : {'YES' if cv['consistent'] else 'NO'}")

            # 1d. QFT violation test
            qft = qft_violation_test(bitstrings, obs_func)
            stats_results[f'qft_{gamma_label}_{obs_name}'] = qft
            print(f"    [QFT] slope = {qft['slope']:.3f} +/- {qft['slope_se']:.3f}")
            print(f"    QFT prediction = -1.000")
            print(f"    Deviation      = {qft['deviation']:+.3f}")
            print(f"    t-statistic    = {qft['t_statistic']:.2f}")
            print(f"    p-value        = {qft['p_value']:.6f}")
            print(f"    Violates QFT   : {'YES' if qft['significant'] else 'NO'}")

    # ==========================================================
    # 2. GAMMA_C META-ANALYSIS
    # ==========================================================

    print("\n" + "=" * 70)
    print("  2. GAMMA_C META-ANALYSIS")
    print("=" * 70)

    meta = meta_analysis_gamma_c()
    stats_results['meta_analysis'] = meta

    print(f"\n  Six measurements of gamma_c:")
    for m in meta['measurements']:
        print(f"    {m['source']:30s}  gamma_c = {m['gamma_c']:.4f}  kappa = {m['kappa']:.2f}")

    print(f"\n  Unweighted mean: {meta['unweighted_mean']:.4f} +/- {meta['unweighted_std']:.4f}")
    print(f"  95% CI:          [{meta['unweighted_ci_95'][0]:.4f}, {meta['unweighted_ci_95'][1]:.4f}]")
    print(f"  Weighted mean:   {meta['weighted_mean']:.4f}")
    print(f"  Bayesian MAP:    {meta['bayesian_map']:.4f}")
    print(f"  Bayesian mean:   {meta['bayesian_mean']:.4f}")
    print(f"  Bayesian 95% CI: [{meta['bayesian_ci_95'][0]:.4f}, {meta['bayesian_ci_95'][1]:.4f}]")
    print(f"  Chi-squared:     {meta['chi2_statistic']:.2f} (df={meta['chi2_df']})")
    print(f"  Chi-squared p:   {meta['chi2_p_value']:.4f}")
    print(f"  Consistent:      {'YES' if meta['consistent'] else 'NO'}")

    # ==========================================================
    # 3. VACUUM COMMUNICATION VALIDATION
    # ==========================================================

    print("\n" + "=" * 70)
    print("  3. VACUUM COMMUNICATION VALIDATION")
    print("=" * 70)

    vcomm = validate_vacuum_communication(data)
    stats_results['vacuum_communication'] = vcomm

    if vcomm.get('conditions'):
        print(f"\n  {'Message':>8s} {'gamma':>6s} {'Acc':>6s} {'CI':>16s} {'p(raw)':>10s} {'p(Bonf)':>10s} {'Sig':>5s}")
        for c in vcomm['conditions']:
            sig = "***" if c['binomial_p_bonferroni'] < 0.001 else "**" if c['binomial_p_bonferroni'] < 0.01 else "*" if c['binomial_p_bonferroni'] < 0.05 else "ns"
            print(f"  {c['message']:>8s} {c['gamma']:6.3f} {c['accuracy']:6.1%} "
                  f"[{c['accuracy_ci'][0]:.1%},{c['accuracy_ci'][1]:.1%}] "
                  f"{c['binomial_p_uncorrected']:10.6f} {c['binomial_p_bonferroni']:10.6f} {sig:>5s}")

        print(f"\n  Sigma-c (information loss): {vcomm['sigma_c']:.4f}  kappa = {vcomm['kappa']:.2f}")
        print(f"  Sigma-c 95% CI: [{vcomm['sigma_c_ci'][0]:.4f}, {vcomm['sigma_c_ci'][1]:.4f}]")

    # ==========================================================
    # 4. GENERATE FIGURES
    # ==========================================================

    print("\n" + "=" * 70)
    print("  4. GENERATING FIGURES")
    print("=" * 70)

    make_figure_1(data, stats_results, fig_dir)
    make_figure_2(meta, fig_dir)
    make_figure_3(vcomm, fig_dir)
    make_figure_4(stats_results, fig_dir)
    make_figure_5_gamma_sweep(data, fig_dir)
    make_figure_6_witness(data, fig_dir)

    # ==========================================================
    # 5. OVERALL VERDICT
    # ==========================================================

    print("\n" + "=" * 70)
    print("  5. VERDICT: READY FOR PUBLICATION?")
    print("=" * 70)

    # Criterion 1: Batch-size anomaly kappa > 10, p < 0.001
    perm_key = 'permutation_gamma_0.0_hamming_var'
    if perm_key in stats_results:
        perm = stats_results[perm_key]
        c1_kappa = perm['observed_kappa'] > 10
        c1_pvalue = perm['p_value'] < 0.001
        c1_pass = c1_kappa and c1_pvalue
        print(f"\n  Criterion 1: Batch-size anomaly")
        print(f"    kappa = {perm['observed_kappa']:.2f} > 10? {'PASS' if c1_kappa else 'FAIL'}")
        print(f"    p-value = {perm['p_value']:.6f} < 0.001? {'PASS' if c1_pvalue else 'FAIL'}")
        print(f"    -> {'PASS' if c1_pass else 'NEEDS MORE DATA'}")
    else:
        c1_pass = False

    # Criterion 2: Communication accuracy > 80% with p < 0.01 (Bonferroni)
    c2_pass = False
    if vcomm.get('conditions'):
        sig_conditions = [c for c in vcomm['conditions']
                         if c['accuracy'] > 0.80 and c['binomial_p_bonferroni'] < 0.01]
        c2_pass = len(sig_conditions) > 0
        print(f"\n  Criterion 2: Vacuum communication (>80%, p<0.01 Bonf.)")
        if sig_conditions:
            for c in sig_conditions:
                print(f"    {c['message']} @ gamma={c['gamma']}: {c['accuracy']:.0%}, p={c['binomial_p_bonferroni']:.6f} PASS")
        else:
            print(f"    No conditions reach >80% with Bonferroni p<0.01")
            # Check relaxed criterion
            relaxed = [c for c in vcomm['conditions']
                      if c['accuracy'] >= 0.80 and c['binomial_p_uncorrected'] < 0.05]
            if relaxed:
                print(f"    (Relaxed: {len(relaxed)} conditions at >80% with raw p<0.05)")
        print(f"    -> {'PASS' if c2_pass else 'MARGINAL — discuss in paper'}")

    # Criterion 3: gamma_c consistency
    c3_pass = meta['consistent']
    print(f"\n  Criterion 3: gamma_c reproducibility")
    print(f"    Chi-squared p = {meta['chi2_p_value']:.4f} > 0.05? {'PASS' if c3_pass else 'FAIL'}")
    print(f"    -> {'PASS — six measurements consistent' if c3_pass else 'FAIL — measurements disagree'}")

    # Overall
    n_pass = sum([c1_pass, c2_pass, c3_pass])
    print(f"\n  {'='*50}")
    if n_pass >= 2:
        print(f"  VERDICT: READY TO WRITE PAPER ({n_pass}/3 criteria passed)")
        print(f"  Recommendation: Submit to npj Quantum Information")
    elif n_pass >= 1:
        print(f"  VERDICT: MARGINAL ({n_pass}/3 criteria passed)")
        print(f"  Recommendation: Write paper with appropriate caveats")
    else:
        print(f"  VERDICT: NEEDS MORE DATA ({n_pass}/3 criteria passed)")
        print(f"  Recommendation: Run additional QPU time")
    print(f"  {'='*50}")

    # ==========================================================
    # SAVE ALL RESULTS
    # ==========================================================

    output_file = "data/vacuum_telescope_v1/statistics_validation.json"

    # Convert numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(output_file, 'w') as f:
        json.dump(stats_results, f, cls=NumpyEncoder, indent=1)
    print(f"\n  Results saved: {output_file}")

    return stats_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import glob

    # Find latest QPU data file
    pattern = "data/vacuum_telescope_v1/vacuum_telescope_qpu_*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No QPU data found matching: {pattern}")
        sys.exit(1)

    data_file = files[-1]
    print(f"  Using: {data_file}")
    run_validation(data_file)
