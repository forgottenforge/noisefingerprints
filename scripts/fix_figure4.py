#!/usr/bin/env python3
"""Regenerate Figure 4: variance scaling log-log plot with correct 'i.i.d.' label."""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

DATA_FILE = "data/vacuum_telescope_v1/vacuum_telescope_qpu_20260131_133417.json"
OUT_FILE = "data/vacuum_telescope_v1/figures/figure_4_variance_scaling.png"


def hamming_weights(bitstrings):
    return np.array([sum(int(b) for b in bs) for bs in bitstrings])


def compute_variance_scaling(bitstrings):
    n = len(bitstrings)
    batch_sizes = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400]
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
    return log_k, log_v, slope, se, intercept


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    block = data['blocks']['V1_z_sweep']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, gamma_target in enumerate([0.0, 0.67]):
        ax = axes[ax_idx]

        # Find closest gamma
        best_m = None
        best_dist = 999
        for m in block['measurements']:
            d = abs(m['gamma'] - gamma_target)
            if d < best_dist:
                best_dist = d
                best_m = m

        bitstrings = best_m['bitstrings']
        gamma_actual = best_m['gamma']
        print(f"gamma={gamma_target}: using gamma={gamma_actual}, {len(bitstrings)} shots")

        log_k, log_v, slope, se, intercept = compute_variance_scaling(bitstrings)

        # Data points
        ax.plot(log_k, log_v, 'ko', ms=8, zorder=5, label='Data')

        # Fit line
        fit_x = np.linspace(log_k.min(), log_k.max(), 100)
        ax.plot(fit_x, slope * fit_x + intercept,
                'C0-', lw=2, label=rf'Fit: $\alpha = {slope:.2f} \pm {se:.2f}$')

        # i.i.d. prediction (NOT "QFT")
        iid_intercept = log_v.mean() - (-1.0) * log_k.mean()
        ax.plot(fit_x, -1.0 * fit_x + iid_intercept,
                'r--', lw=2, label=r'i.i.d.: $\alpha = -1.0$')

        ax.set_xlabel(r'$\ln(K)$', fontsize=12)
        ax.set_ylabel(r'$\ln(\mathrm{Var})$', fontsize=12)
        ax.set_title(rf'Variance scaling: $\gamma = {gamma_actual}$', fontsize=12)
        ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
