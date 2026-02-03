#!/usr/bin/env python3
"""Regenerate Figure 8: cross-platform variance scaling with correct 'i.i.d.' label."""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

RIGETTI_FILE = "data/vacuum_telescope_v1/replication_rigetti_qpu_20260201_100409.json"
IONQ_FILE = "data/vacuum_telescope_v1/replication_ionq_qpu_20260201_095758.json"
OUT_FILE = "data/vacuum_telescope_v1/figures/figure_8_variance_cross.png"


def hamming_weights(bitstrings):
    return np.array([sum(int(b) for b in bs) for bs in bitstrings])


def load_bitstrings(filepath, block_name, gamma_target):
    with open(filepath) as f:
        data = json.load(f)
    block = data['blocks'][block_name]
    for m in block['measurements']:
        if abs(m['gamma'] - gamma_target) < 0.01:
            return m['bitstrings']
    return None


def variance_scaling(bitstrings):
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
    return log_k, log_v, slope, se


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, gamma in enumerate([0.0, 0.67]):
        ax = axes[ax_idx]

        # Rigetti
        rig_bs = load_bitstrings(RIGETTI_FILE, "rigetti_boost", gamma)
        if rig_bs:
            log_k, log_v, slope, se = variance_scaling(rig_bs)
            ax.plot(log_k, log_v, 'o', color='C0', ms=6,
                    label=rf'Rigetti ($\alpha = {slope:.2f}$)')
            print(f"Rigetti gamma={gamma}: alpha={slope:.2f} +/- {se:.2f}, N={len(rig_bs)}")

        # IonQ
        ionq_bs = load_bitstrings(IONQ_FILE, "R23_batch", gamma)
        if ionq_bs:
            log_k, log_v, slope, se = variance_scaling(ionq_bs)
            ax.plot(log_k, log_v, 's', color='C1', ms=6,
                    label=rf'IonQ ($\alpha = {slope:.2f}$)')
            print(f"IonQ gamma={gamma}: alpha={slope:.2f} +/- {se:.2f}, N={len(ionq_bs)}")

        # i.i.d. prediction (NOT "QFT")
        ax.plot([1, 7], [-1*1+2, -1*7+2], 'r--', lw=1.5, label=r'i.i.d.: $\alpha = -1.0$')

        ax.set_xlabel(r'$\ln(K)$', fontsize=12)
        ax.set_ylabel(r'$\ln(\mathrm{Var})$', fontsize=12)
        ax.set_title(rf'$\gamma = {gamma}$', fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle('Variance scaling: Rigetti vs IonQ', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
