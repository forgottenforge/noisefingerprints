#!/usr/bin/env python3
"""
VACUUM TELESCOPE — CROSS-PLATFORM REPLICATION
===============================================

Phase 1: Rigetti Ankaa-3 boost (10,000 shots at 4 gamma values)
Phase 2: IonQ Forte-1 full replication (12-point sweep + 2×2000 batch analysis)

Budget: EUR 100 total
  Rigetti:  4 tasks × 10,000 shots = $15.20 (EUR 14.07)
  IonQ:    28 tasks ×  8,700 shots = $92.60 (EUR 85.74)
  Total:   32 tasks, ~$107.80 (EUR 99.81)

Purpose: Make paper airtight with:
  1. 10,000-shot batch analysis (vs original 1,200) — Rigetti
  2. Independent QPU confirmation — IonQ trapped ions
  3. Cross-platform gamma_c comparison

USAGE:
  python vacuum_replication.py --phase rigetti   # Run Rigetti boost
  python vacuum_replication.py --phase ionq      # Run IonQ replication
  python vacuum_replication.py --phase sim       # Test both on simulator
  python vacuum_replication.py --analyze         # Cross-platform analysis

Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import argparse
import json
import time
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

try:
    from braket.circuits import Circuit
    from braket.aws import AwsDevice
    from braket.devices import LocalSimulator
except ImportError:
    print("[ERROR] Amazon Braket SDK required: pip install amazon-braket-sdk")
    sys.exit(1)

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
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'figure.dpi': 300,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
    })
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("data/vacuum_telescope_v1")
N_QUBITS = 6
DEPHASING_FACTOR = 1.5
DAMPING_FACTOR = 0.5
EUR_TO_USD = 1.08

# Device configs
DEVICES = {
    'rigetti': {
        'arn': 'arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3',
        'region': 'us-west-1',
        'cost_task': 0.30,
        'cost_shot': 0.00035,
        'label': 'Rigetti Ankaa-3 (superconducting)',
    },
    'ionq': {
        'arn': 'arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1',
        'region': 'us-east-1',
        'cost_task': 0.30,
        'cost_shot': 0.01,
        'label': 'IonQ Forte-1 (trapped ion)',
    },
}

# --- Rigetti boost: 4 gammas × 10,000 shots ---
RIGETTI_GAMMAS = [0.0, 0.67, 1.0]
RIGETTI_SHOTS = 10000

# --- IonQ replication ---
# R1: gamma sweep (12 points, concentrated near gamma_c)
IONQ_SWEEP_GAMMAS = [0.0, 0.15, 0.35, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.88, 0.94, 1.0]
IONQ_SWEEP_SHOTS = 250

# R2/R3: high-shot batch analysis
IONQ_BATCH_GAMMAS = [0.0, 0.67]
IONQ_BATCH_SHOTS = 2000

# R4: product state control
IONQ_CONTROL_GAMMAS = [0.0, 1.0]
IONQ_CONTROL_SHOTS = 250

# R5: vacuum communication
IONQ_COMM_MESSAGES = ["101010", "111000", "110011", "100110"]
IONQ_COMM_GAMMAS = [0.0, 0.67, 1.0]
IONQ_COMM_SHOTS = 100


# =============================================================================
# REPLICATION ENGINE
# =============================================================================

class ReplicationEngine:

    def __init__(self, device_key: str, mode: str = 'qpu'):
        self.device_key = device_key
        self.mode = mode
        self.spent_usd = 0.0
        self.task_count = 0
        self.total_shots = 0

        cfg = DEVICES[device_key]
        self.cost_task = cfg['cost_task']
        self.cost_shot = cfg['cost_shot']
        self.label = cfg['label']

        if mode == 'qpu':
            print(f"  Connecting to {self.label}...")
            self.device = AwsDevice(cfg['arn'])
            print(f"  Connected.")
        else:
            print(f"  Simulating {self.label}")
            self.device = LocalSimulator("braket_dm")

        self.data = {
            'metadata': {
                'experiment': 'vacuum_telescope_replication',
                'timestamp_start': datetime.now().isoformat(),
                'device': device_key,
                'device_label': self.label,
                'mode': mode,
                'n_qubits': N_QUBITS,
                'noise_model': {
                    'dephasing_factor': DEPHASING_FACTOR,
                    'damping_factor': DAMPING_FACTOR,
                },
            },
            'blocks': {}
        }

    def _cost(self, n_tasks, shots):
        if self.mode != 'qpu':
            return 0.0
        return n_tasks * (self.cost_task + shots * self.cost_shot)

    def _record(self, shots):
        if self.mode == 'qpu':
            self.spent_usd += self.cost_task + shots * self.cost_shot
        self.task_count += 1
        self.total_shots += shots

    # --- Noise injection (same as vacuum_telescope.py) ---

    def _add_dephasing(self, circuit, qubits, gamma, seed=42):
        rng = np.random.RandomState(seed)
        for q in qubits:
            if rng.random() < gamma:
                angle = rng.uniform(-np.pi * gamma, np.pi * gamma)
                circuit.rz(q, angle)
        return circuit

    def _add_amplitude_damping(self, circuit, qubits, gamma, seed=42):
        rng = np.random.RandomState(seed)
        for q in qubits:
            if rng.random() < gamma:
                circuit.rx(q, -gamma * np.pi)
        return circuit

    # --- Circuit builders ---

    def build_chain(self, gamma, seed=42):
        circuit = Circuit()
        circuit.h(0)
        for i in range(N_QUBITS - 1):
            circuit.cnot(i, i + 1)
            if gamma > 0:
                self._add_dephasing(circuit, [i, i + 1],
                                    gamma * DEPHASING_FACTOR, seed=seed + i * 100)
                self._add_amplitude_damping(circuit, [i, i + 1],
                                            gamma * DAMPING_FACTOR, seed=seed + i * 100 + 50)
        return circuit

    def build_product(self):
        circuit = Circuit()
        for q in range(N_QUBITS):
            circuit.h(q)
        return circuit

    def build_communication(self, message, gamma, seed=42):
        circuit = Circuit()
        for i, bit in enumerate(message):
            if bit == '1':
                circuit.x(i)
        circuit.h(0)
        for i in range(N_QUBITS - 1):
            circuit.cnot(i, i + 1)
            if gamma > 0:
                s = int(gamma * 10000)
                self._add_dephasing(circuit, [i, i + 1],
                                    gamma * DEPHASING_FACTOR, seed=s + i * 100)
                self._add_amplitude_damping(circuit, [i, i + 1],
                                            gamma * DAMPING_FACTOR, seed=s + i * 100 + 50)
        return circuit

    # --- Measurement ---

    def run_z(self, circuit, shots):
        circ = circuit.copy()
        for q in range(N_QUBITS):
            circ.measure(q)
        task = self.device.run(circ, shots=int(shots))
        result = task.result()
        bitstrings = [''.join(str(int(b)) for b in row) for row in result.measurements]
        self._record(shots)
        return bitstrings

    # --- Bitstring statistics ---

    @staticmethod
    def hamming_weights(bitstrings):
        return np.array([sum(int(b) for b in bs) for bs in bitstrings])

    @staticmethod
    def entropy(bitstrings):
        counts = Counter(bitstrings)
        total = len(bitstrings)
        probs = np.array([c / total for c in counts.values()])
        return float(-np.sum(probs * np.log2(probs + 1e-15)))

    # =========================================================================
    # RIGETTI BOOST
    # =========================================================================

    def run_rigetti_boost(self):
        print("\n" + "=" * 70)
        print("  RIGETTI BOOST — 10,000-shot batch analysis")
        print("=" * 70)
        est = self._cost(len(RIGETTI_GAMMAS), RIGETTI_SHOTS)
        print(f"  {len(RIGETTI_GAMMAS)} gammas × {RIGETTI_SHOTS:,} shots")
        print(f"  Estimated cost: ${est:.2f} (EUR {est/EUR_TO_USD:.2f})")

        block = {
            'description': 'High-shot batch analysis for statistical power',
            'shots': RIGETTI_SHOTS,
            'gamma_values': RIGETTI_GAMMAS,
            'measurements': [],
        }

        for idx, gamma in enumerate(RIGETTI_GAMMAS):
            t0 = time.time()
            circuit = self.build_chain(gamma, seed=42)
            bitstrings = self.run_z(circuit, RIGETTI_SHOTS)
            dt = time.time() - t0
            hw = self.hamming_weights(bitstrings)
            ent = self.entropy(bitstrings)
            n_unique = len(set(bitstrings))

            block['measurements'].append({
                'gamma': gamma,
                'bitstrings': bitstrings,
                'n_shots': len(bitstrings),
                'timestamp': datetime.now().isoformat(),
                'elapsed_s': round(dt, 2),
            })
            print(f"  [{idx+1}/{len(RIGETTI_GAMMAS)}] gamma={gamma:.2f}  "
                  f"unique={n_unique:4d}/{2**N_QUBITS}  "
                  f"<H>={np.mean(hw):.2f}  S={ent:.2f}  ({dt:.1f}s)")

        block['status'] = 'COMPLETE'
        self.data['blocks']['rigetti_boost'] = block

    # =========================================================================
    # IONQ REPLICATION
    # =========================================================================

    def run_ionq_replication(self):
        print("\n" + "=" * 70)
        print("  IONQ FORTE-1 — FULL REPLICATION")
        print("=" * 70)

        # --- R1: Gamma sweep ---
        est_r1 = self._cost(len(IONQ_SWEEP_GAMMAS), IONQ_SWEEP_SHOTS)
        print(f"\n  [R1] Gamma sweep: {len(IONQ_SWEEP_GAMMAS)} gammas × {IONQ_SWEEP_SHOTS} shots")
        print(f"       Estimated: ${est_r1:.2f}")

        r1 = {'description': 'Gamma sweep on IonQ', 'measurements': []}
        for idx, gamma in enumerate(IONQ_SWEEP_GAMMAS):
            t0 = time.time()
            circuit = self.build_chain(gamma, seed=42)
            bitstrings = self.run_z(circuit, IONQ_SWEEP_SHOTS)
            dt = time.time() - t0
            hw = self.hamming_weights(bitstrings)
            n_unique = len(set(bitstrings))

            r1['measurements'].append({
                'gamma': gamma,
                'bitstrings': bitstrings,
                'n_shots': len(bitstrings),
                'timestamp': datetime.now().isoformat(),
            })
            print(f"       [{idx+1}/{len(IONQ_SWEEP_GAMMAS)}] gamma={gamma:.2f}  "
                  f"unique={n_unique:3d}  <H>={np.mean(hw):.2f}  ({dt:.1f}s)")

        r1['status'] = 'COMPLETE'
        self.data['blocks']['R1_sweep'] = r1
        if self.mode == 'qpu':
            print(f"       Cost so far: ${self.spent_usd:.2f}")

        # --- R2/R3: High-shot batch analysis ---
        est_r23 = self._cost(len(IONQ_BATCH_GAMMAS), IONQ_BATCH_SHOTS)
        print(f"\n  [R2-R3] Batch analysis: {len(IONQ_BATCH_GAMMAS)} gammas × {IONQ_BATCH_SHOTS:,} shots")
        print(f"          Estimated: ${est_r23:.2f}")

        r23 = {'description': 'High-shot batch analysis on IonQ', 'measurements': []}
        for idx, gamma in enumerate(IONQ_BATCH_GAMMAS):
            t0 = time.time()
            circuit = self.build_chain(gamma, seed=42)
            bitstrings = self.run_z(circuit, IONQ_BATCH_SHOTS)
            dt = time.time() - t0
            hw = self.hamming_weights(bitstrings)
            n_unique = len(set(bitstrings))

            r23['measurements'].append({
                'gamma': gamma,
                'bitstrings': bitstrings,
                'n_shots': len(bitstrings),
                'timestamp': datetime.now().isoformat(),
            })
            print(f"          gamma={gamma:.2f}  unique={n_unique:4d}  "
                  f"<H>={np.mean(hw):.2f}  ({dt:.1f}s)")

        r23['status'] = 'COMPLETE'
        self.data['blocks']['R23_batch'] = r23
        if self.mode == 'qpu':
            print(f"          Cost so far: ${self.spent_usd:.2f}")

        # --- R4: Product state control ---
        est_r4 = self._cost(len(IONQ_CONTROL_GAMMAS), IONQ_CONTROL_SHOTS)
        print(f"\n  [R4] Product state control: {len(IONQ_CONTROL_GAMMAS)} gammas × {IONQ_CONTROL_SHOTS} shots")
        print(f"       Estimated: ${est_r4:.2f}")

        r4 = {'description': 'Product state control on IonQ', 'measurements': []}
        for gamma in IONQ_CONTROL_GAMMAS:
            t0 = time.time()
            circuit = self.build_product()
            bitstrings = self.run_z(circuit, IONQ_CONTROL_SHOTS)
            dt = time.time() - t0
            n_unique = len(set(bitstrings))

            r4['measurements'].append({
                'gamma': gamma,
                'bitstrings': bitstrings,
                'n_shots': len(bitstrings),
                'timestamp': datetime.now().isoformat(),
            })
            print(f"       gamma={gamma:.2f}  unique={n_unique:3d}  ({dt:.1f}s)")

        r4['status'] = 'COMPLETE'
        self.data['blocks']['R4_control'] = r4
        if self.mode == 'qpu':
            print(f"       Cost so far: ${self.spent_usd:.2f}")

        # --- R5: Vacuum communication ---
        n_tasks = len(IONQ_COMM_MESSAGES) * len(IONQ_COMM_GAMMAS)
        est_r5 = self._cost(n_tasks, IONQ_COMM_SHOTS)
        print(f"\n  [R5] Communication: {len(IONQ_COMM_MESSAGES)} msgs × {len(IONQ_COMM_GAMMAS)} gammas × {IONQ_COMM_SHOTS} shots")
        print(f"       Estimated: ${est_r5:.2f}")

        r5 = {'description': 'Vacuum communication on IonQ', 'measurements': []}
        for msg in IONQ_COMM_MESSAGES:
            print(f"\n       Message: {msg}")
            for gamma in IONQ_COMM_GAMMAS:
                t0 = time.time()
                circuit = self.build_communication(msg, gamma, seed=42)
                bitstrings = self.run_z(circuit, IONQ_COMM_SHOTS)
                dt = time.time() - t0

                decoded_bits = []
                qubit_probs = []
                for qi in range(N_QUBITS):
                    ones = sum(int(bs[qi]) for bs in bitstrings)
                    prob = ones / len(bitstrings)
                    qubit_probs.append(prob)
                    decoded_bits.append('1' if prob > 0.5 else '0')
                decoded = ''.join(decoded_bits)
                accuracy = sum(a == b for a, b in zip(msg, decoded)) / len(msg)

                r5['measurements'].append({
                    'message': msg,
                    'gamma': gamma,
                    'decoded': decoded,
                    'accuracy': accuracy,
                    'qubit_probabilities': qubit_probs,
                    'bitstrings': bitstrings,
                    'n_shots': len(bitstrings),
                    'timestamp': datetime.now().isoformat(),
                })

                match = "OK" if accuracy >= 0.83 else "LOST" if accuracy <= 0.5 else "~"
                print(f"         gamma={gamma:.3f}  {msg} -> {decoded}  "
                      f"acc={accuracy:.0%}  [{match}]  ({dt:.1f}s)")

        r5['status'] = 'COMPLETE'
        self.data['blocks']['R5_communication'] = r5
        if self.mode == 'qpu':
            print(f"       Cost so far: ${self.spent_usd:.2f}")

    # =========================================================================
    # SAVE
    # =========================================================================

    def save(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = str(OUTPUT_DIR / f"replication_{self.device_key}_{self.mode}_{ts}.json")

        self.data['metadata']['timestamp_end'] = datetime.now().isoformat()
        self.data['metadata']['total_tasks'] = self.task_count
        self.data['metadata']['total_shots'] = self.total_shots
        self.data['metadata']['total_cost_usd'] = round(self.spent_usd, 2)
        self.data['metadata']['total_cost_eur'] = round(self.spent_usd / EUR_TO_USD, 2)

        class NE(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(fname, 'w') as f:
            json.dump(self.data, f, cls=NE, indent=1)
        size_mb = Path(fname).stat().st_size / (1024 * 1024)
        print(f"\n  Data saved: {fname} ({size_mb:.1f} MB)")
        return fname


# =============================================================================
# CROSS-PLATFORM ANALYSIS
# =============================================================================

def compute_sigma_c(sigmas, observables, sw=5):
    if len(observables) < sw:
        obs_smooth = observables
    else:
        obs_smooth = savgol_filter(observables, sw, min(3, sw-1), mode='nearest')
    chi = np.abs(np.gradient(obs_smooth, sigmas))
    peaks, _ = find_peaks(chi, prominence=0.001)
    if len(peaks) == 0:
        pi = np.argmax(chi)
        sc = sigmas[pi]
        bl = np.median(chi) + 1e-15
        kp = chi[pi] / bl
    else:
        proms = peak_prominences(chi, peaks)[0]
        pi = peaks[np.argmax(proms)]
        sc = sigmas[pi]
        bl = np.median(chi) + 1e-15
        kp = proms.max() / bl
    return chi, float(sc), float(kp)


def hamming_weights(bs):
    return np.array([sum(int(b) for b in s) for s in bs])


def entropy(bs):
    c = Counter(bs)
    n = len(bs)
    p = np.array([v/n for v in c.values()])
    return float(-np.sum(p * np.log2(p + 1e-15)))


def batch_analysis(bitstrings, label=""):
    n = len(bitstrings)
    batch_sizes = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
    batch_sizes = [k for k in batch_sizes if k <= n // 3]

    results = {}
    obs_funcs = {
        'hamming_mean': lambda bs: float(np.mean(hamming_weights(bs))),
        'hamming_var': lambda bs: float(np.var(hamming_weights(bs))),
        'entropy': lambda bs: entropy(bs),
    }

    for obs_name, obs_func in obs_funcs.items():
        ks, means, stds = [], [], []
        for K in batch_sizes:
            nb = n // K
            vals = [obs_func(bitstrings[b*K:(b+1)*K]) for b in range(nb)]
            ks.append(K)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        ks_arr = np.array(ks, dtype=float)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        chi, sc, kp = compute_sigma_c(ks_arr, means_arr)

        # Variance scaling
        variances = stds_arr ** 2
        valid = variances > 0
        if valid.sum() >= 3:
            slope, _, _, _, se = stats.linregress(np.log(ks_arr[valid]),
                                                   np.log(variances[valid] + 1e-30))
            t_stat = (slope - (-1.0)) / (se + 1e-15)
            p_val = 2 * stats.t.sf(abs(t_stat), df=valid.sum() - 2)
        else:
            slope, se, t_stat, p_val = -1.0, 999.0, 0.0, 1.0

        results[obs_name] = {
            'batch_sizes': [int(k) for k in ks],
            'batch_means': means,
            'batch_stds': [float(s) for s in stds_arr],
            'chi': chi.tolist(),
            'sigma_c': sc,
            'kappa': kp,
            'var_slope': slope,
            'var_slope_se': se,
            'qft_t_stat': t_stat,
            'qft_p_value': p_val,
        }

    return results


def run_cross_platform_analysis():
    """Analyze and compare Rigetti (original + boost) and IonQ data."""
    import glob

    print("=" * 70)
    print("  CROSS-PLATFORM ANALYSIS")
    print("=" * 70)

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load all data files
    datasets = {}

    # Original Rigetti
    orig_files = sorted(glob.glob(str(OUTPUT_DIR / "vacuum_telescope_qpu_*.json")))
    if orig_files:
        with open(orig_files[-1]) as f:
            datasets['rigetti_original'] = json.load(f)
        print(f"  Loaded Rigetti original: {orig_files[-1]}")

    # Rigetti boost
    boost_files = sorted(glob.glob(str(OUTPUT_DIR / "replication_rigetti_qpu_*.json")))
    if boost_files:
        with open(boost_files[-1]) as f:
            datasets['rigetti_boost'] = json.load(f)
        print(f"  Loaded Rigetti boost: {boost_files[-1]}")

    # IonQ
    ionq_files = sorted(glob.glob(str(OUTPUT_DIR / "replication_ionq_qpu_*.json")))
    if not ionq_files:
        ionq_files = sorted(glob.glob(str(OUTPUT_DIR / "replication_ionq_sim_*.json")))
    if ionq_files:
        with open(ionq_files[-1]) as f:
            datasets['ionq'] = json.load(f)
        print(f"  Loaded IonQ: {ionq_files[-1]}")

    all_results = {}

    # ---- 1. BATCH-SIZE ANALYSIS (Rigetti boost) ----
    print("\n" + "-" * 70)
    print("  1. BATCH-SIZE ANALYSIS (10,000 shots)")
    print("-" * 70)

    if 'rigetti_boost' in datasets:
        boost = datasets['rigetti_boost']
        for meas in boost['blocks'].get('rigetti_boost', {}).get('measurements', []):
            gamma = meas['gamma']
            bs = meas['bitstrings']
            print(f"\n  Rigetti 10k shots @ gamma={gamma:.2f} ({len(bs)} shots)")
            results = batch_analysis(bs)
            for obs, r in results.items():
                qft_sig = "***" if r['qft_p_value'] < 0.001 else "**" if r['qft_p_value'] < 0.01 else "*" if r['qft_p_value'] < 0.05 else "ns"
                print(f"    {obs:15s}  sigma_c={r['sigma_c']:6.0f}  kappa={r['kappa']:7.2f}  "
                      f"slope={r['var_slope']:+.3f}±{r['var_slope_se']:.3f}  "
                      f"t={r['qft_t_stat']:+.2f}  p={r['qft_p_value']:.6f}  {qft_sig}")
            all_results[f'rigetti_boost_g{gamma}'] = results

    # ---- 2. IONQ GAMMA SWEEP ----
    print("\n" + "-" * 70)
    print("  2. IONQ GAMMA SWEEP")
    print("-" * 70)

    if 'ionq' in datasets:
        ionq = datasets['ionq']
        sweep = ionq['blocks'].get('R1_sweep', {})
        if sweep.get('measurements'):
            gammas, entropies, mean_h, std_h = [], [], [], []
            for meas in sweep['measurements']:
                bs = meas['bitstrings']
                hw = hamming_weights(bs)
                gammas.append(meas['gamma'])
                entropies.append(entropy(bs))
                mean_h.append(float(np.mean(hw)))
                std_h.append(float(np.std(hw)))

            gammas_arr = np.array(gammas)
            chi_e, sc_e, kp_e = compute_sigma_c(gammas_arr, np.array(entropies))
            chi_h, sc_h, kp_h = compute_sigma_c(gammas_arr, np.array(mean_h))

            print(f"  Entropy:      sigma_c = {sc_e:.4f}  kappa = {kp_e:.2f}")
            print(f"  Mean Hamming: sigma_c = {sc_h:.4f}  kappa = {kp_h:.2f}")

            all_results['ionq_sweep'] = {
                'gammas': gammas, 'entropies': entropies,
                'mean_hamming': mean_h, 'std_hamming': std_h,
                'sigma_c_entropy': sc_e, 'kappa_entropy': kp_e,
                'sigma_c_hamming': sc_h, 'kappa_hamming': kp_h,
            }

        # ---- 3. IONQ BATCH ANALYSIS ----
        print("\n" + "-" * 70)
        print("  3. IONQ BATCH-SIZE ANALYSIS (2,000 shots)")
        print("-" * 70)

        batch_block = ionq['blocks'].get('R23_batch', {})
        for meas in batch_block.get('measurements', []):
            gamma = meas['gamma']
            bs = meas['bitstrings']
            print(f"\n  IonQ 2k shots @ gamma={gamma:.2f} ({len(bs)} shots)")
            results = batch_analysis(bs)
            for obs, r in results.items():
                qft_sig = "***" if r['qft_p_value'] < 0.001 else "**" if r['qft_p_value'] < 0.01 else "*" if r['qft_p_value'] < 0.05 else "ns"
                print(f"    {obs:15s}  sigma_c={r['sigma_c']:6.0f}  kappa={r['kappa']:7.2f}  "
                      f"slope={r['var_slope']:+.3f}±{r['var_slope_se']:.3f}  "
                      f"t={r['qft_t_stat']:+.2f}  p={r['qft_p_value']:.6f}  {qft_sig}")
            all_results[f'ionq_batch_g{gamma}'] = results

        # ---- 4. IONQ COMMUNICATION ----
        print("\n" + "-" * 70)
        print("  4. IONQ VACUUM COMMUNICATION")
        print("-" * 70)

        comm_block = ionq['blocks'].get('R5_communication', {})
        if comm_block.get('measurements'):
            print(f"\n  {'Message':>8s} {'gamma':>6s} {'Acc':>6s} {'Decoded':>8s}")
            for m in comm_block['measurements']:
                print(f"  {m['message']:>8s} {m['gamma']:6.3f} {m['accuracy']:6.0%} {m['decoded']:>8s}")

    # ---- 5. CROSS-PLATFORM COMPARISON ----
    print("\n" + "-" * 70)
    print("  5. CROSS-PLATFORM COMPARISON")
    print("-" * 70)

    # Compare variance slopes
    print(f"\n  {'Platform':12s} {'gamma':>6s} {'Observable':>15s} {'slope':>8s} {'SE':>8s} {'p(QFT)':>10s}")
    print(f"  {'-'*65}")

    for key, results in sorted(all_results.items()):
        if 'batch' in key or 'boost' in key:
            platform = 'Rigetti' if 'rigetti' in key else 'IonQ'
            gamma = key.split('_g')[-1] if '_g' in key else '?'
            for obs, r in results.items():
                if obs == 'hamming_var':
                    sig = "***" if r['qft_p_value'] < 0.001 else "**" if r['qft_p_value'] < 0.01 else "*" if r['qft_p_value'] < 0.05 else "ns"
                    print(f"  {platform:12s} {gamma:>6s} {obs:>15s} {r['var_slope']:>+8.3f} {r['var_slope_se']:>8.3f} {r['qft_p_value']:>10.6f} {sig}")

    # ---- 6. GAMMA_C COMPARISON ----
    if 'ionq_sweep' in all_results:
        ionq_sc = all_results['ionq_sweep']
        print(f"\n  gamma_c comparison:")
        print(f"    Rigetti (E3):  0.674  (kappa=8.58)")
        print(f"    Rigetti (V1):  0.710  (kappa=6.35)")
        print(f"    IonQ entropy:  {ionq_sc['sigma_c_entropy']:.4f}  (kappa={ionq_sc['kappa_entropy']:.2f})")
        print(f"    IonQ Hamming:  {ionq_sc['sigma_c_hamming']:.4f}  (kappa={ionq_sc['kappa_hamming']:.2f})")

        # Test if consistent
        all_gammas = [0.674, 0.684, 0.71, 0.58, 0.674, 0.62,
                      ionq_sc['sigma_c_entropy'], ionq_sc['sigma_c_hamming']]
        mean_gc = np.mean(all_gammas)
        std_gc = np.std(all_gammas, ddof=1)
        print(f"\n    8-measurement mean: {mean_gc:.4f} ± {std_gc:.4f}")

    # Save
    output = str(OUTPUT_DIR / "cross_platform_analysis.json")
    class NE(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output, 'w') as f:
        json.dump(all_results, f, cls=NE, indent=1)
    print(f"\n  Results saved: {output}")

    # ---- FIGURES ----
    if HAS_PLOT and 'ionq_sweep' in all_results:
        make_cross_platform_figures(datasets, all_results, fig_dir)


def make_cross_platform_figures(datasets, results, fig_dir):
    """Generate cross-platform comparison figures."""

    # Figure 7: Rigetti vs IonQ gamma sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Rigetti original
    rig = datasets.get('rigetti_original', {})
    v1 = rig.get('blocks', {}).get('V1_z_sweep', {})
    if v1.get('measurements'):
        rg, re = [], []
        for m in v1['measurements']:
            rg.append(m['gamma'])
            re.append(entropy(m['bitstrings']))
        ax1.plot(rg, re, 'o-', color='C0', ms=4, label='Rigetti Ankaa-3')

    # IonQ
    ionq = datasets.get('ionq', {})
    r1 = ionq.get('blocks', {}).get('R1_sweep', {})
    if r1.get('measurements'):
        ig, ie = [], []
        for m in r1['measurements']:
            ig.append(m['gamma'])
            ie.append(entropy(m['bitstrings']))
        ax1.plot(ig, ie, 's-', color='C1', ms=6, label='IonQ Forte-1')

    ax1.axvline(0.674, color='red', ls='--', alpha=0.5, label=r'$\gamma_c$')
    ax1.set_xlabel(r'$\gamma$')
    ax1.set_ylabel('Shannon entropy (bits)')
    ax1.set_title('(a) Bitstring entropy')
    ax1.legend()

    # Hamming distance
    if v1.get('measurements'):
        rh = [float(np.mean(hamming_weights(m['bitstrings']))) for m in v1['measurements']]
        ax2.plot(rg, rh, 'o-', color='C0', ms=4, label='Rigetti')
    if r1.get('measurements'):
        ih = [float(np.mean(hamming_weights(m['bitstrings']))) for m in r1['measurements']]
        ax2.plot(ig, ih, 's-', color='C1', ms=6, label='IonQ')

    ax2.axvline(0.674, color='red', ls='--', alpha=0.5)
    ax2.set_xlabel(r'$\gamma$')
    ax2.set_ylabel(r'$\langle H \rangle$')
    ax2.set_title('(b) Mean Hamming weight')
    ax2.legend()

    fig.suptitle('Cross-platform comparison: Rigetti vs IonQ', fontsize=14)
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure_7_cross_platform.png')
    plt.close()
    print(f"  Saved: {fig_dir / 'figure_7_cross_platform.png'}")

    # Figure 8: Variance scaling comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, gamma in enumerate([0.0, 0.67]):
        ax = axes[ax_idx]
        for platform, color, marker in [('rigetti_boost', 'C0', 'o'), ('ionq_batch', 'C1', 's')]:
            key = f'{platform}_g{gamma}'
            if key in results and 'hamming_var' in results[key]:
                r = results[key]['hamming_var']
                ks = np.array(r['batch_sizes'], dtype=float)
                stds = np.array(r['batch_stds'])
                v = stds ** 2
                valid = v > 0
                label_name = 'Rigetti' if 'rigetti' in platform else 'IonQ'
                ax.plot(np.log(ks[valid]), np.log(v[valid]), marker,
                        color=color, ms=6, label=f'{label_name} ($\\alpha={r["var_slope"]:.2f}$)')

        # QFT prediction
        ax.plot([1, 7], [-1*1+2, -1*7+2], 'r--', lw=1.5, label=r'QFT: $\alpha=-1.0$')
        ax.set_xlabel(r'$\ln(K)$')
        ax.set_ylabel(r'$\ln(\mathrm{Var})$')
        ax.set_title(f'$\\gamma = {gamma}$')
        ax.legend(fontsize=9)

    fig.suptitle('Variance scaling: Rigetti vs IonQ', fontsize=14)
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure_8_variance_cross.png')
    plt.close()
    print(f"  Saved: {fig_dir / 'figure_8_variance_cross.png'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vacuum Telescope Replication')
    parser.add_argument('--phase', choices=['rigetti', 'ionq', 'sim', 'both'],
                        default='sim', help='Which QPU to run')
    parser.add_argument('--analyze', action='store_true', help='Run cross-platform analysis')
    args = parser.parse_args()

    if args.analyze:
        run_cross_platform_analysis()
        sys.exit(0)

    if args.phase == 'sim':
        # Test both on simulator
        print("=" * 70)
        print("  SIMULATOR TEST — Rigetti boost + IonQ replication")
        print("=" * 70)

        eng = ReplicationEngine('rigetti', mode='sim')
        eng.run_rigetti_boost()
        fname = eng.save()
        print(f"\n  Rigetti sim: {eng.task_count} tasks, {eng.total_shots:,} shots")

        eng2 = ReplicationEngine('ionq', mode='sim')
        eng2.run_ionq_replication()
        fname2 = eng2.save()
        print(f"\n  IonQ sim: {eng2.task_count} tasks, {eng2.total_shots:,} shots")

        print("\n  Running cross-platform analysis on sim data...")
        run_cross_platform_analysis()

    elif args.phase == 'rigetti':
        eng = ReplicationEngine('rigetti', mode='qpu')
        eng.run_rigetti_boost()
        eng.save()
        print(f"\n  Total: {eng.task_count} tasks, {eng.total_shots:,} shots, "
              f"${eng.spent_usd:.2f} (EUR {eng.spent_usd/EUR_TO_USD:.2f})")

    elif args.phase == 'ionq':
        eng = ReplicationEngine('ionq', mode='qpu')
        eng.run_ionq_replication()
        eng.save()
        print(f"\n  Total: {eng.task_count} tasks, {eng.total_shots:,} shots, "
              f"${eng.spent_usd:.2f} (EUR {eng.spent_usd/EUR_TO_USD:.2f})")

    elif args.phase == 'both':
        print("=" * 70)
        print("  RUNNING BOTH QPUs")
        print("=" * 70)

        eng = ReplicationEngine('rigetti', mode='qpu')
        eng.run_rigetti_boost()
        eng.save()
        print(f"\n  Rigetti: {eng.task_count} tasks, ${eng.spent_usd:.2f}")

        eng2 = ReplicationEngine('ionq', mode='qpu')
        eng2.run_ionq_replication()
        eng2.save()
        total = eng.spent_usd + eng2.spent_usd
        print(f"\n  IonQ: {eng2.task_count} tasks, ${eng2.spent_usd:.2f}")
        print(f"\n  TOTAL: ${total:.2f} (EUR {total/EUR_TO_USD:.2f})")

        print("\n  Running cross-platform analysis...")
        run_cross_platform_analysis()
