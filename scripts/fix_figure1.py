#!/usr/bin/env python3
"""Regenerate Figure 1: batch-size susceptibility chi(K) for the PRA paper.

Matches the computation in vacuum_statistics.py:
  - Observable = Hamming-weight variance (computed within each batch)
  - chi(K) = |d<O>/dK| where <O> = mean of per-batch observable values
  - Savitzky-Golay smoothing (window=5, polyorder=3)
  - kappa from scipy peak_prominences
"""

import json
import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter

DATA_FILE = "data/vacuum_telescope_v1/vacuum_telescope_qpu_20260131_133417.json"
OUT_FILE = "data/vacuum_telescope_v1/figures/figure_1_chi_K.png"


def hamming_weights(bitstrings):
    return np.array([sum(int(b) for b in bs) for bs in bitstrings])


def compute_sigma_c(sigmas, observables, smoothing_window=5):
    """Exact copy of vacuum_statistics.py compute_sigma_c."""
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


def batch_sigma_c_from_bitstrings(bitstrings, obs_func):
    """Exact copy of vacuum_statistics.py batch_sigma_c_from_bitstrings."""
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

    return batch_ks, batch_means_arr, batch_stds_arr, chi, sigma_c, kappa


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    block = data['blocks']['V1_z_sweep']
    bitstrings_g0 = None
    for m in block['measurements']:
        if abs(m['gamma'] - 0.0) < 0.01:
            bitstrings_g0 = m['bitstrings']
            break

    if bitstrings_g0 is None:
        print("ERROR: gamma=0 data not found")
        return

    print(f"V1 gamma=0: {len(bitstrings_g0)} shots")

    # Observable: Hamming-weight variance within each batch
    obs_func = lambda bs: float(np.var(hamming_weights(bs)))
    ks, means, stds, chi, sigma_c, kappa = batch_sigma_c_from_bitstrings(bitstrings_g0, obs_func)

    print(f"sigma_c = {sigma_c:.0f}, kappa = {kappa:.2f}")
    print(f"Batch sizes: {ks.tolist()}")
    print(f"Chi values: {[f'{c:.4f}' for c in chi]}")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(ks, chi, 'o-', color='C0', lw=2, ms=6, label=r'$\chi(K)$ for Var($H$)')

    # Mark sigma_c
    sc_idx = np.argmin(np.abs(ks - sigma_c))
    ax.plot(sigma_c, chi[sc_idx], 's', color='red', ms=12, zorder=5,
            label=rf'$\sigma_c = {sigma_c:.0f}$ ($\kappa = {kappa:.1f}$)')
    ax.axvline(sigma_c, color='red', ls=':', alpha=0.4)

    ax.set_xlabel('Batch size $K$', fontsize=12)
    ax.set_ylabel(r'$\chi(K) = |d\langle O \rangle / dK|$', fontsize=12)
    ax.set_title(r'Batch-size susceptibility: Rigetti Ankaa-3, $\gamma = 0$', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
