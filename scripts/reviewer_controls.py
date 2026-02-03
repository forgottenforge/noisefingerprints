#!/usr/bin/env python3
"""
reviewer_controls.py — Address reviewer-requested controls:
  1. Shuffled-shots control: does alpha change when shot order is randomized?
  2. Downsample control: does Rigetti alpha remain stable at N=2000 (matching IonQ)?

Both use existing QPU data — no new experiments required.
"""

import json
import numpy as np
from scipy import stats
from collections import Counter
import os

DATA_DIR = "data/vacuum_telescope_v1"
RIGETTI_FILE = os.path.join(DATA_DIR, "replication_rigetti_qpu_20260201_100409.json")
IONQ_FILE = os.path.join(DATA_DIR, "replication_ionq_qpu_20260201_095758.json")

N_SHUFFLES = 1000
N_DOWNSAMPLES = 200
RNG = np.random.default_rng(42)


def hamming_weights(bitstrings):
    return np.array([sum(int(b) for b in bs) for bs in bitstrings])


def bitstring_entropy(bitstrings):
    counts = Counter(bitstrings)
    total = len(bitstrings)
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-15)))


def compute_alpha(bitstrings, obs_name="hamming_var"):
    """Compute variance scaling exponent alpha from bitstrings."""
    n = len(bitstrings)
    batch_sizes = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
    batch_sizes = [k for k in batch_sizes if k <= n // 3]

    obs_funcs = {
        'hamming_mean': lambda bs: np.mean(hamming_weights(bs)),
        'hamming_var': lambda bs: np.var(hamming_weights(bs)),
        'entropy': lambda bs: bitstring_entropy(bs),
    }
    obs_func = obs_funcs[obs_name]

    ks, variances = [], []
    for K in batch_sizes:
        nb = n // K
        vals = [obs_func(bitstrings[b * K:(b + 1) * K]) for b in range(nb)]
        if len(vals) >= 3:
            ks.append(K)
            variances.append(np.var(vals))

    ks = np.array(ks, dtype=float)
    variances = np.array(variances)
    valid = variances > 0

    if valid.sum() >= 3:
        slope, intercept, r, p, se = stats.linregress(
            np.log(ks[valid]), np.log(variances[valid] + 1e-30)
        )
        t_stat = (slope - (-1.0)) / (se + 1e-15)
        p_val = 2 * stats.t.sf(abs(t_stat), df=valid.sum() - 2)
        return {'alpha': slope, 'se': se, 't_stat': t_stat, 'p_value': p_val}
    return {'alpha': np.nan, 'se': np.nan, 't_stat': np.nan, 'p_value': np.nan}


def load_bitstrings(filepath, block_name, gamma_target):
    """Load bitstrings for a specific gamma from a block."""
    with open(filepath) as f:
        data = json.load(f)
    block = data['blocks'][block_name]
    for m in block['measurements']:
        if abs(m['gamma'] - gamma_target) < 0.01:
            return m['bitstrings']
    raise ValueError(f"gamma={gamma_target} not found in {block_name}")


def shuffled_shots_control(bitstrings, label, obs_name="hamming_var"):
    """Control 1: Shuffle shot order and recompute alpha."""
    n = len(bitstrings)
    print(f"\n  {label}: {n} shots, observable={obs_name}")

    # Original alpha (preserving temporal order)
    orig = compute_alpha(bitstrings, obs_name)
    print(f"    Original alpha  = {orig['alpha']:.3f} +/- {orig['se']:.3f}  (p={orig['p_value']:.2e})")

    # Shuffled alpha distribution
    shuffled_alphas = []
    for i in range(N_SHUFFLES):
        idx = RNG.permutation(n)
        shuffled = [bitstrings[j] for j in idx]
        result = compute_alpha(shuffled, obs_name)
        if not np.isnan(result['alpha']):
            shuffled_alphas.append(result['alpha'])

    shuffled_alphas = np.array(shuffled_alphas)
    mean_s = np.mean(shuffled_alphas)
    std_s = np.std(shuffled_alphas)
    ci_lo = np.percentile(shuffled_alphas, 2.5)
    ci_hi = np.percentile(shuffled_alphas, 97.5)

    # How many shuffled runs still deviate from -1?
    n_deviate = np.sum(np.abs(shuffled_alphas - (-1.0)) > np.abs(orig['alpha'] - (-1.0)))
    p_shuffle = n_deviate / len(shuffled_alphas)

    print(f"    Shuffled alpha  = {mean_s:.3f} +/- {std_s:.3f}  [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"    Shift from orig = {mean_s - orig['alpha']:+.3f}")
    print(f"    Shuffle p-value = {p_shuffle:.4f}  (fraction with |dev| >= original)")

    if abs(mean_s - (-1.0)) < 2 * std_s:
        print(f"    --> Shuffled alpha CONSISTENT with i.i.d. (alpha=-1.0)")
        print(f"        => Original deviation is TEMPORAL (shot-order dependent)")
    else:
        print(f"    --> Shuffled alpha STILL DEVIATES from -1.0")
        print(f"        => Deviation is DISTRIBUTIONAL (not purely temporal)")

    return {
        'original_alpha': orig['alpha'],
        'original_se': orig['se'],
        'original_p': orig['p_value'],
        'shuffled_mean': float(mean_s),
        'shuffled_std': float(std_s),
        'shuffled_ci': [float(ci_lo), float(ci_hi)],
        'shuffle_p': float(p_shuffle),
        'n_shuffles': N_SHUFFLES,
    }


def downsample_control(bitstrings_full, label, target_n=2000, obs_name="hamming_var"):
    """Control 2: Downsample to target_n shots, compare alpha."""
    n = len(bitstrings_full)
    if n <= target_n:
        print(f"\n  {label}: Only {n} shots, cannot downsample to {target_n}")
        return None

    print(f"\n  {label}: Downsampling {n} -> {target_n} shots ({N_DOWNSAMPLES} iterations)")

    # Full dataset alpha
    full = compute_alpha(bitstrings_full, obs_name)
    print(f"    Full ({n} shots) alpha = {full['alpha']:.3f} +/- {full['se']:.3f}")

    # Downsampled alpha distribution (contiguous blocks to preserve temporal structure)
    down_alphas = []
    for i in range(N_DOWNSAMPLES):
        # Random contiguous block of target_n shots
        start = RNG.integers(0, n - target_n + 1)
        subset = bitstrings_full[start:start + target_n]
        result = compute_alpha(subset, obs_name)
        if not np.isnan(result['alpha']):
            down_alphas.append(result['alpha'])

    down_alphas = np.array(down_alphas)
    mean_d = np.mean(down_alphas)
    std_d = np.std(down_alphas)
    ci_lo = np.percentile(down_alphas, 2.5)
    ci_hi = np.percentile(down_alphas, 97.5)

    print(f"    Down ({target_n} shots) alpha = {mean_d:.3f} +/- {std_d:.3f}  [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"    Shift from full = {mean_d - full['alpha']:+.3f}")

    # Is full alpha within downsample CI?
    covers = ci_lo <= full['alpha'] <= ci_hi
    print(f"    Full alpha in downsample 95% CI: {'YES' if covers else 'NO'}")

    return {
        'full_n': n,
        'full_alpha': full['alpha'],
        'full_se': full['se'],
        'target_n': target_n,
        'downsample_mean': float(mean_d),
        'downsample_std': float(std_d),
        'downsample_ci': [float(ci_lo), float(ci_hi)],
        'full_in_ci': bool(covers),
        'n_iterations': N_DOWNSAMPLES,
    }


def main():
    print("=" * 70)
    print("  REVIEWER CONTROLS — Shuffled-Shots & Downsample Analysis")
    print("=" * 70)

    results = {}

    # --- Load data ---
    rig_g0 = load_bitstrings(RIGETTI_FILE, "rigetti_boost", 0.0)
    rig_g67 = load_bitstrings(RIGETTI_FILE, "rigetti_boost", 0.67)
    ionq_g0 = load_bitstrings(IONQ_FILE, "R23_batch", 0.0)
    ionq_g67 = load_bitstrings(IONQ_FILE, "R23_batch", 0.67)

    print(f"\n  Data loaded:")
    print(f"    Rigetti boost: {len(rig_g0)} shots (gamma=0), {len(rig_g67)} shots (gamma=0.67)")
    print(f"    IonQ batch:    {len(ionq_g0)} shots (gamma=0), {len(ionq_g67)} shots (gamma=0.67)")

    # ================================================================
    # CONTROL 1: Shuffled shots
    # ================================================================
    print("\n" + "=" * 70)
    print("  CONTROL 1: SHUFFLED-SHOTS (temporal vs. distributional)")
    print("=" * 70)

    for obs in ['hamming_var', 'hamming_mean', 'entropy']:
        print(f"\n  --- Observable: {obs} ---")
        results[f'shuffle_rigetti_g0_{obs}'] = shuffled_shots_control(rig_g0, "Rigetti gamma=0", obs)
        results[f'shuffle_rigetti_g67_{obs}'] = shuffled_shots_control(rig_g67, "Rigetti gamma=0.67", obs)
        results[f'shuffle_ionq_g0_{obs}'] = shuffled_shots_control(ionq_g0, "IonQ gamma=0", obs)
        results[f'shuffle_ionq_g67_{obs}'] = shuffled_shots_control(ionq_g67, "IonQ gamma=0.67", obs)

    # ================================================================
    # CONTROL 2: Downsample Rigetti to N=2000
    # ================================================================
    print("\n" + "=" * 70)
    print("  CONTROL 2: DOWNSAMPLE (alpha stability across N)")
    print("=" * 70)

    for obs in ['hamming_var', 'hamming_mean', 'entropy']:
        print(f"\n  --- Observable: {obs} ---")
        results[f'downsample_rigetti_g0_{obs}'] = downsample_control(rig_g0, "Rigetti gamma=0", 2000, obs)
        results[f'downsample_rigetti_g67_{obs}'] = downsample_control(rig_g67, "Rigetti gamma=0.67", 2000, obs)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print("\n  Shuffled-Shots Control (hamming_var):")
    for key in ['shuffle_rigetti_g0_hamming_var', 'shuffle_rigetti_g67_hamming_var',
                'shuffle_ionq_g0_hamming_var', 'shuffle_ionq_g67_hamming_var']:
        r = results[key]
        label = key.replace('shuffle_', '').replace('_hamming_var', '')
        print(f"    {label:25s}: orig={r['original_alpha']:+.3f}  shuffled={r['shuffled_mean']:+.3f} +/- {r['shuffled_std']:.3f}")

    print("\n  Downsample Control (hamming_var):")
    for key in ['downsample_rigetti_g0_hamming_var', 'downsample_rigetti_g67_hamming_var']:
        r = results[key]
        if r is None:
            continue
        label = key.replace('downsample_', '').replace('_hamming_var', '')
        print(f"    {label:25s}: full(10k)={r['full_alpha']:+.3f}  down(2k)={r['downsample_mean']:+.3f} +/- {r['downsample_std']:.3f}  in_CI={'yes' if r['full_in_ci'] else 'NO'}")

    # Save results
    outfile = os.path.join(DATA_DIR, "reviewer_controls.json")
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {outfile}")


if __name__ == "__main__":
    main()
