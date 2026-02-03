#!/usr/bin/env python3
"""
VACUUM TELESCOPE v1.0
=====================
Purpose-designed experiment for detecting quantum vacuum structure
via sigma-c susceptibility analysis on shot-level statistics.

Runs on Rigetti Ankaa-3 via Amazon Braket.
Budget: EUR 50 (~88 tasks, ~65,000 shots)

DESIGN PRINCIPLE: Store ALL raw bitstrings. NEVER aggregate prematurely.
Every shot is a pixel in the vacuum photograph.

BLOCKS:
  V1 - Gamma sweep (Z-basis, entangled chain): 30 gammas x 1200 shots
       -> Bitstring mining: entropy, kurtosis, patterns at each gamma
  V2 - Witness calibration (XX/YY bases): 16 gammas x 400 shots
       -> Entanglement witness reconstruction (ZZ from V1, XX/YY from V2)
  V3 - Temporal structure (repeated identical runs): 8 x 1000 shots
       -> Run-to-run correlations, vacuum memory test
  V4 - Control (product state, no entanglement): 3 x 500 shots
       -> Null control: sigma-c should find NOTHING
  V5 - GHZ comparison (maximally fragile state): 5 x 1000 shots
       -> Independent probe of same vacuum transition
  V6 - Depth sweep (NATURAL QPU decoherence): 10 depths x 1500 shots
       -> Real vacuum coupling (no artificial noise injection)
  V7 - Vacuum communication: 4 messages x 10 gammas x 500 shots
       -> Information survival through decoherence at gamma_c

POST-HOC ANALYSIS (from raw bitstrings):
  - Batch-size sigma-c (scale structure of measurement statistics)
  - Bitstring entropy saturation curve
  - Hamming weight distribution vs gamma
  - Temporal autocorrelation between runs
  - QFT consistency test (expected vs observed distributions)
  - Forbidden pattern detection
  - Cross-circuit comparison (chain vs GHZ vs product)

USAGE:
  python vacuum_telescope.py --mode sim     # Test on local simulator
  python vacuum_telescope.py --mode qpu     # Run on Rigetti Ankaa-3
  python vacuum_telescope.py --analyze FILE # Analyze collected data

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
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

try:
    from braket.circuits import Circuit
    from braket.aws import AwsDevice
    from braket.devices import LocalSimulator
    HAS_BRAKET = True
except ImportError:
    HAS_BRAKET = False
    print("[WARNING] Amazon Braket SDK not installed. Install with: pip install amazon-braket-sdk")

try:
    from scipy import stats
    from scipy.signal import savgol_filter, find_peaks, peak_prominences
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("data/vacuum_telescope_v1")

# Budget (V1-V6: ~$50, V7: ~$10, total ~$60 = EUR ~55)
BUDGET_EUR = 55.0
COST_PER_TASK = 0.30    # USD per Braket task
COST_PER_SHOT = 0.00035  # USD per shot on Rigetti
EUR_TO_USD = 1.08

# Device
DEVICE_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"

# Qubit count
N_QUBITS = 6

# Noise model (matches E3 from magnetguy paper)
# E3 used dephasing_factor=1.5, damping_factor=0.5
# E6 used dephasing_factor=0.6, damping_factor=0.4
# We use E3 model for direct comparison with gamma_c = 0.6737
DEPHASING_FACTOR = 1.5
DAMPING_FACTOR = 0.5

# --- V1: Gamma sweep (Z-basis, main experiment) ---
# Non-uniform grid: coarse far from gamma_c, fine near gamma_c = 0.6737
_V1_FAR = [0.0, 0.08, 0.16, 0.25, 0.35, 0.45, 0.85, 1.0]        # 8 points
_V1_NEAR = [0.50, 0.54, 0.58, 0.76, 0.80]                         # 5 points
_V1_CRITICAL = list(np.linspace(0.59, 0.76, 18))                   # 18 points (step ~0.01)
V1_GAMMAS = sorted(set(round(g, 6) for g in _V1_FAR + _V1_NEAR + _V1_CRITICAL))
V1_SHOTS = 1200

# --- V2: Witness calibration (XX/YY bases) ---
# 10 strategic gammas (concentrated around gamma_c, saves 12 tasks vs 16)
V2_GAMMAS = [0.0, 0.50, 0.60, 0.64, 0.674, 0.70, 0.74, 0.80, 0.90, 1.0]
V2_SHOTS = 400  # per basis (XX or YY)
V2_QUBIT_PAIR = (2, 3)  # middle of chain for best signal

# --- V3: Temporal structure ---
V3_GAMMA = 0.674    # right at gamma_c
V3_N_RUNS = 8
V3_SHOTS = 1000

# --- V4: Control (product state) ---
V4_GAMMAS = [0.0, 0.50, 1.0]
V4_SHOTS = 500

# --- V5: GHZ comparison ---
V5_GAMMAS = [0.0, 0.50, 0.674, 0.80, 1.0]
V5_SHOTS = 1000

# --- V6: Depth sweep (natural decoherence, NO injected noise) ---
V6_DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
V6_SHOTS = 1500

# --- V7: Vacuum communication (information survival test) ---
# Encode known bit patterns in initial state, entangle, decohere, try to recover.
# Tests: does gamma_c mark the boundary where encoded information is lost?
V7_MESSAGES = [
    "101010",  # alternating
    "111000",  # block
    "110011",  # pattern
    "100110",  # pseudo-random
]
# 6 strategic gammas: before, at, and after gamma_c
V7_GAMMAS = [0.0, 0.40, 0.62, 0.674, 0.80, 1.0]
V7_SHOTS = 500


# =============================================================================
# VACUUM TELESCOPE
# =============================================================================

class VacuumTelescope:
    """Quantum vacuum structure experiment on Rigetti Ankaa-3."""

    def __init__(self, mode: str = 'sim', budget_eur: float = BUDGET_EUR):
        self.mode = mode
        self.budget_eur = budget_eur
        self.spent_usd = 0.0
        self.task_count = 0
        self.total_shots = 0

        if not HAS_BRAKET:
            raise RuntimeError("Amazon Braket SDK required. pip install amazon-braket-sdk")

        if mode == 'qpu':
            print(f"  Connecting to Rigetti Ankaa-3...")
            try:
                self.device = AwsDevice(DEVICE_ARN)
                print(f"  Connected. Budget: EUR {budget_eur:.2f}")
            except Exception as e:
                print(f"  QPU connection failed: {e}")
                print(f"  Falling back to density-matrix simulator")
                self.mode = 'sim'
                self.device = LocalSimulator("braket_dm")
        else:
            print(f"  Using local density-matrix simulator")
            self.device = LocalSimulator("braket_dm")

        self.data = {
            'metadata': {
                'experiment': 'vacuum_telescope_v1',
                'timestamp_start': datetime.now().isoformat(),
                'device': self.mode,
                'n_qubits': N_QUBITS,
                'noise_model': {
                    'dephasing_factor': DEPHASING_FACTOR,
                    'damping_factor': DAMPING_FACTOR,
                    'description': 'Matches E3 from magnetguy paper'
                },
                'gamma_c_prior': 0.6737,
                'budget_eur': budget_eur,
            },
            'blocks': {}
        }

    # -------------------------------------------------------------------------
    # Budget tracking
    # -------------------------------------------------------------------------

    def _estimate_cost(self, n_tasks: int, shots_per_task: int) -> float:
        if self.mode != 'qpu':
            return 0.0
        return n_tasks * (COST_PER_TASK + shots_per_task * COST_PER_SHOT)

    def _check_budget(self, cost_usd: float, label: str = "") -> bool:
        if self.mode != 'qpu':
            return True
        budget_usd = self.budget_eur * EUR_TO_USD
        remaining = budget_usd - self.spent_usd
        if cost_usd > remaining * 1.05:  # 5% tolerance
            print(f"  [BUDGET] {label}: need ${cost_usd:.2f}, have ${remaining:.2f} -- SKIPPING")
            return False
        return True

    def _record_cost(self, shots: int):
        if self.mode == 'qpu':
            self.spent_usd += COST_PER_TASK + shots * COST_PER_SHOT
        self.task_count += 1
        self.total_shots += shots

    # -------------------------------------------------------------------------
    # Circuit builders
    # -------------------------------------------------------------------------

    def _build_entangled_chain(self, gamma: float, seed: int = 42) -> Circuit:
        """
        6-qubit CNOT cascade with controlled decoherence.
        Identical to E3 from magnetguy paper.

        H(0) -> CNOT(0,1) -> CNOT(1,2) -> ... -> CNOT(4,5)
        With dephasing and amplitude damping after each CNOT.
        """
        circuit = Circuit()
        circuit.h(0)
        for i in range(N_QUBITS - 1):
            circuit.cnot(i, i + 1)
            if gamma > 0:
                self._add_dephasing(circuit, [i, i + 1], gamma * DEPHASING_FACTOR, seed=seed + i * 100)
                self._add_amplitude_damping(circuit, [i, i + 1], gamma * DAMPING_FACTOR, seed=seed + i * 100 + 50)
        return circuit

    def _build_ghz_state(self, gamma: float, seed: int = 42) -> Circuit:
        """
        GHZ state: (|000000> + |111111>)/sqrt(2) with controlled decoherence.
        Matches E6 structure but with E3 noise model.
        """
        circuit = Circuit()
        circuit.h(0)
        for i in range(1, N_QUBITS):
            circuit.cnot(0, i)
        if gamma > 0:
            self._add_dephasing(circuit, list(range(N_QUBITS)),
                                gamma * DEPHASING_FACTOR, seed=seed)
            self._add_amplitude_damping(circuit, list(range(N_QUBITS)),
                                        gamma * DAMPING_FACTOR, seed=seed + 50)
        return circuit

    def _build_product_state(self) -> Circuit:
        """
        Product state |++++++> (no entanglement).
        Control experiment: sigma-c should find NO critical scale.
        """
        circuit = Circuit()
        for q in range(N_QUBITS):
            circuit.h(q)
        return circuit

    def _build_deep_chain(self, n_layers: int) -> Circuit:
        """
        Multi-layer CNOT cascade WITHOUT injected noise.
        Natural QPU decoherence increases with depth.
        Each layer: CNOT(0,1), CNOT(1,2), ..., CNOT(4,5)
        """
        circuit = Circuit()
        circuit.h(0)
        for layer in range(n_layers):
            for i in range(N_QUBITS - 1):
                circuit.cnot(i, i + 1)
        return circuit

    # -------------------------------------------------------------------------
    # Noise injection (identical to magnetguy2.py)
    # -------------------------------------------------------------------------

    def _add_dephasing(self, circuit: Circuit, qubits: List[int],
                       gamma: float, seed: int = 42) -> Circuit:
        """Approximate dephasing via random RZ rotations."""
        rng = np.random.RandomState(seed)
        for q in qubits:
            if rng.random() < gamma:
                angle = rng.uniform(-np.pi * gamma, np.pi * gamma)
                circuit.rz(q, angle)
        return circuit

    def _add_amplitude_damping(self, circuit: Circuit, qubits: List[int],
                               gamma: float, seed: int = 42) -> Circuit:
        """Approximate T1 decay via RX rotation."""
        rng = np.random.RandomState(seed)
        for q in qubits:
            if rng.random() < gamma:
                circuit.rx(q, -gamma * np.pi)
        return circuit

    # -------------------------------------------------------------------------
    # Measurement helpers
    # -------------------------------------------------------------------------

    def _run_z_basis(self, circuit: Circuit, shots: int) -> List[str]:
        """
        Run Z-basis measurement. Returns list of raw bitstrings.
        Each bitstring is a string of '0' and '1', length N_QUBITS.
        ORDER: bitstring[0] = qubit 0, bitstring[1] = qubit 1, etc.
        """
        circ = circuit.copy()
        for q in range(N_QUBITS):
            circ.measure(q)

        task = self.device.run(circ, shots=int(shots))
        result = task.result()

        # Extract raw measurements (numpy array of shape [shots, n_qubits])
        measurements = result.measurements

        # Convert to bitstrings, preserving qubit order
        bitstrings = []
        for row in measurements:
            bs = ''.join(str(int(b)) for b in row)
            bitstrings.append(bs)

        self._record_cost(shots)
        return bitstrings

    def _run_pauli_basis(self, circuit: Circuit, q1: int, q2: int,
                         basis: str, shots: int) -> List[str]:
        """
        Run measurement in specified Pauli basis for a qubit pair.
        basis: 'XX', 'YY', or 'ZZ'
        Returns raw bitstrings (only the 2 measured qubits).
        """
        circ = circuit.copy()

        if basis == 'XX':
            circ.h(q1)
            circ.h(q2)
        elif basis == 'YY':
            circ.rx(q1, np.pi / 2)
            circ.rx(q2, np.pi / 2)
        # ZZ: no rotation needed

        circ.measure(q1)
        circ.measure(q2)

        task = self.device.run(circ, shots=int(shots))
        result = task.result()

        measurements = result.measurements

        bitstrings = []
        for row in measurements:
            bs = ''.join(str(int(b)) for b in row)
            bitstrings.append(bs)

        self._record_cost(shots)
        return bitstrings

    # -------------------------------------------------------------------------
    # V1: Gamma sweep (Z-basis, entangled chain)
    # -------------------------------------------------------------------------

    def V1_gamma_sweep(self) -> Dict:
        """
        Primary experiment: fine gamma sweep in Z basis.
        Stores ALL raw bitstrings for post-hoc analysis.
        """
        n_tasks = len(V1_GAMMAS)
        cost = self._estimate_cost(n_tasks, V1_SHOTS)
        print(f"\n  [V1] Gamma sweep: {n_tasks} gammas x {V1_SHOTS} shots")
        print(f"       Gamma range: {V1_GAMMAS[0]:.3f} to {V1_GAMMAS[-1]:.3f}")
        print(f"       Critical region: 18 points in [0.59, 0.76] (step ~0.01)")
        print(f"       Estimated cost: ${cost:.2f}")

        if not self._check_budget(cost, "V1"):
            return {'status': 'SKIPPED'}

        block_data = {
            'description': 'Z-basis gamma sweep for bitstring mining',
            'circuit_type': 'entangled_chain',
            'measurement_basis': 'Z',
            'n_qubits': N_QUBITS,
            'shots_per_gamma': V1_SHOTS,
            'noise_model': 'E3',
            'gamma_values': V1_GAMMAS,
            'measurements': []
        }

        for idx, gamma in enumerate(V1_GAMMAS):
            t0 = time.time()
            circuit = self._build_entangled_chain(gamma, seed=int(gamma * 10000))
            bitstrings = self._run_z_basis(circuit, V1_SHOTS)
            dt = time.time() - t0

            entry = {
                'gamma': gamma,
                'bitstrings': bitstrings,
                'counts': dict(Counter(bitstrings)),
                'n_shots': len(bitstrings),
                'timestamp': datetime.now().isoformat(),
                'elapsed_s': round(dt, 2),
                'seed': int(gamma * 10000),
            }
            block_data['measurements'].append(entry)

            # Quick stats for progress display
            n_unique = len(set(bitstrings))
            hamming_mean = np.mean([sum(int(b) for b in bs) for bs in bitstrings])
            in_critical = 0.59 <= gamma <= 0.76
            marker = " *" if in_critical else ""
            print(f"       [{idx+1}/{n_tasks}] gamma={gamma:.4f}  "
                  f"unique={n_unique:3d}/{2**N_QUBITS}  "
                  f"<H>={hamming_mean:.2f}  "
                  f"({dt:.1f}s){marker}")

        block_data['status'] = 'COMPLETE'
        return block_data

    # -------------------------------------------------------------------------
    # V2: Witness calibration (XX/YY bases)
    # -------------------------------------------------------------------------

    def V2_witness_calibration(self) -> Dict:
        """
        XX and YY measurements for entanglement witness reconstruction.
        ZZ comes for free from V1 Z-basis data.
        W = (XX + YY)/2 - |ZZ| for qubit pair (2,3).
        """
        n_tasks = len(V2_GAMMAS) * 2  # XX + YY per gamma
        cost = self._estimate_cost(n_tasks, V2_SHOTS)
        print(f"\n  [V2] Witness calibration: {len(V2_GAMMAS)} gammas x 2 bases x {V2_SHOTS} shots")
        print(f"       Qubit pair: ({V2_QUBIT_PAIR[0]}, {V2_QUBIT_PAIR[1]})")
        print(f"       Estimated cost: ${cost:.2f}")

        if not self._check_budget(cost, "V2"):
            return {'status': 'SKIPPED'}

        q1, q2 = V2_QUBIT_PAIR

        block_data = {
            'description': 'XX/YY measurements for witness reconstruction',
            'circuit_type': 'entangled_chain',
            'qubit_pair': list(V2_QUBIT_PAIR),
            'shots_per_basis': V2_SHOTS,
            'gamma_values': V2_GAMMAS,
            'measurements': []
        }

        for idx, gamma in enumerate(V2_GAMMAS):
            circuit = self._build_entangled_chain(gamma, seed=int(gamma * 10000))

            # XX measurement
            t0 = time.time()
            xx_bitstrings = self._run_pauli_basis(circuit, q1, q2, 'XX', V2_SHOTS)
            dt_xx = time.time() - t0

            # YY measurement
            t1 = time.time()
            yy_bitstrings = self._run_pauli_basis(circuit, q1, q2, 'YY', V2_SHOTS)
            dt_yy = time.time() - t1

            entry = {
                'gamma': gamma,
                'XX': {
                    'bitstrings': xx_bitstrings,
                    'counts': dict(Counter(xx_bitstrings)),
                    'n_shots': len(xx_bitstrings),
                },
                'YY': {
                    'bitstrings': yy_bitstrings,
                    'counts': dict(Counter(yy_bitstrings)),
                    'n_shots': len(yy_bitstrings),
                },
                'timestamp': datetime.now().isoformat(),
                'elapsed_s': round(dt_xx + dt_yy, 2),
                'seed': int(gamma * 10000),
            }
            block_data['measurements'].append(entry)

            # Quick XX/YY expectation
            xx_exp = self._expectation_from_pair_bitstrings(xx_bitstrings)
            yy_exp = self._expectation_from_pair_bitstrings(yy_bitstrings)
            print(f"       [{idx+1}/{len(V2_GAMMAS)}] gamma={gamma:.4f}  "
                  f"<XX>={xx_exp:+.3f}  <YY>={yy_exp:+.3f}  "
                  f"({dt_xx + dt_yy:.1f}s)")

        block_data['status'] = 'COMPLETE'
        return block_data

    def _expectation_from_pair_bitstrings(self, bitstrings: List[str]) -> float:
        """Compute Pauli expectation value from 2-qubit bitstrings."""
        total = len(bitstrings)
        parity_sum = 0
        for bs in bitstrings:
            # Parity: +1 if bits are same, -1 if different
            b0, b1 = int(bs[0]), int(bs[1])
            parity_sum += 1 - 2 * (b0 ^ b1)
        return parity_sum / total

    # -------------------------------------------------------------------------
    # V3: Temporal structure
    # -------------------------------------------------------------------------

    def V3_temporal_structure(self) -> Dict:
        """
        Repeated identical experiments at gamma_c.
        Different seeds per run to get different noise realizations.
        Tests: vacuum memory, device drift, temporal correlations.
        """
        cost = self._estimate_cost(V3_N_RUNS, V3_SHOTS)
        print(f"\n  [V3] Temporal structure: {V3_N_RUNS} runs x {V3_SHOTS} shots at gamma={V3_GAMMA}")
        print(f"       Each run uses a DIFFERENT noise seed")
        print(f"       Estimated cost: ${cost:.2f}")

        if not self._check_budget(cost, "V3"):
            return {'status': 'SKIPPED'}

        block_data = {
            'description': 'Temporal structure: repeated runs at gamma_c',
            'circuit_type': 'entangled_chain',
            'measurement_basis': 'Z',
            'gamma': V3_GAMMA,
            'shots_per_run': V3_SHOTS,
            'n_runs': V3_N_RUNS,
            'measurements': []
        }

        for run_idx in range(V3_N_RUNS):
            # Different seed per run = different noise realization
            seed = 42000 + run_idx * 1000
            t0 = time.time()
            circuit = self._build_entangled_chain(V3_GAMMA, seed=seed)
            bitstrings = self._run_z_basis(circuit, V3_SHOTS)
            dt = time.time() - t0

            entry = {
                'run_index': run_idx,
                'bitstrings': bitstrings,
                'counts': dict(Counter(bitstrings)),
                'n_shots': len(bitstrings),
                'timestamp': datetime.now().isoformat(),
                'elapsed_s': round(dt, 2),
                'seed': seed,
            }
            block_data['measurements'].append(entry)

            n_unique = len(set(bitstrings))
            hamming_mean = np.mean([sum(int(b) for b in bs) for bs in bitstrings])
            print(f"       [run {run_idx+1}/{V3_N_RUNS}] seed={seed}  "
                  f"unique={n_unique:3d}  <H>={hamming_mean:.2f}  ({dt:.1f}s)")

        block_data['status'] = 'COMPLETE'
        return block_data

    # -------------------------------------------------------------------------
    # V4: Control (product state)
    # -------------------------------------------------------------------------

    def V4_control(self) -> Dict:
        """
        Product state |++++++> with no entanglement.
        NULL CONTROL: sigma-c should find NO critical scale.
        If it does, something is wrong with the method.
        """
        cost = self._estimate_cost(len(V4_GAMMAS), V4_SHOTS)
        print(f"\n  [V4] Control (product state): {len(V4_GAMMAS)} gammas x {V4_SHOTS} shots")
        print(f"       Expected: NO critical scale (null control)")
        print(f"       Estimated cost: ${cost:.2f}")

        if not self._check_budget(cost, "V4"):
            return {'status': 'SKIPPED'}

        block_data = {
            'description': 'Product state control (no entanglement)',
            'circuit_type': 'product_state',
            'measurement_basis': 'Z',
            'shots_per_gamma': V4_SHOTS,
            'gamma_values': V4_GAMMAS,
            'measurements': []
        }

        for idx, gamma in enumerate(V4_GAMMAS):
            t0 = time.time()
            # Product state doesn't use gamma — gamma is irrelevant
            # But we inject noise anyway to test whether noise alone creates structure
            circuit = self._build_product_state()
            if gamma > 0:
                self._add_dephasing(circuit, list(range(N_QUBITS)),
                                    gamma * DEPHASING_FACTOR, seed=int(gamma * 10000))
                self._add_amplitude_damping(circuit, list(range(N_QUBITS)),
                                            gamma * DAMPING_FACTOR, seed=int(gamma * 10000) + 50)
            bitstrings = self._run_z_basis(circuit, V4_SHOTS)
            dt = time.time() - t0

            entry = {
                'gamma': gamma,
                'bitstrings': bitstrings,
                'counts': dict(Counter(bitstrings)),
                'n_shots': len(bitstrings),
                'timestamp': datetime.now().isoformat(),
                'elapsed_s': round(dt, 2),
            }
            block_data['measurements'].append(entry)

            n_unique = len(set(bitstrings))
            print(f"       [{idx+1}/{len(V4_GAMMAS)}] gamma={gamma:.3f}  "
                  f"unique={n_unique:3d}  ({dt:.1f}s)")

        block_data['status'] = 'COMPLETE'
        return block_data

    # -------------------------------------------------------------------------
    # V5: GHZ comparison
    # -------------------------------------------------------------------------

    def V5_ghz_comparison(self) -> Dict:
        """
        GHZ state: maximally fragile entanglement.
        Independent probe of the same vacuum transition.
        If gamma_c matches V1 (entangled chain), the threshold is universal.
        """
        cost = self._estimate_cost(len(V5_GAMMAS), V5_SHOTS)
        print(f"\n  [V5] GHZ comparison: {len(V5_GAMMAS)} gammas x {V5_SHOTS} shots")
        print(f"       Expected: gamma_c near 0.67 (matching V1 and E3/E6)")
        print(f"       Estimated cost: ${cost:.2f}")

        if not self._check_budget(cost, "V5"):
            return {'status': 'SKIPPED'}

        block_data = {
            'description': 'GHZ state comparison (maximally fragile)',
            'circuit_type': 'ghz',
            'measurement_basis': 'Z',
            'shots_per_gamma': V5_SHOTS,
            'gamma_values': V5_GAMMAS,
            'measurements': []
        }

        for idx, gamma in enumerate(V5_GAMMAS):
            t0 = time.time()
            circuit = self._build_ghz_state(gamma, seed=int(gamma * 10000))
            bitstrings = self._run_z_basis(circuit, V5_SHOTS)
            dt = time.time() - t0

            entry = {
                'gamma': gamma,
                'bitstrings': bitstrings,
                'counts': dict(Counter(bitstrings)),
                'n_shots': len(bitstrings),
                'timestamp': datetime.now().isoformat(),
                'elapsed_s': round(dt, 2),
                'seed': int(gamma * 10000),
            }
            block_data['measurements'].append(entry)

            # For GHZ, interesting stat: fraction of |000000> and |111111>
            ghz_frac = (bitstrings.count('0' * N_QUBITS) +
                        bitstrings.count('1' * N_QUBITS)) / len(bitstrings)
            print(f"       [{idx+1}/{len(V5_GAMMAS)}] gamma={gamma:.3f}  "
                  f"GHZ_frac={ghz_frac:.3f}  ({dt:.1f}s)")

        block_data['status'] = 'COMPLETE'
        return block_data

    # -------------------------------------------------------------------------
    # V6: Depth sweep (natural QPU decoherence)
    # -------------------------------------------------------------------------

    def V6_depth_sweep(self) -> Dict:
        """
        Multi-layer CNOT cascade WITHOUT injected noise.
        Natural QPU decoherence increases with circuit depth.
        This probes the REAL vacuum coupling (no approximations).

        If sigma-c finds a critical depth, that depth corresponds to
        a natural decoherence threshold set by the actual vacuum environment.
        """
        cost = self._estimate_cost(len(V6_DEPTHS), V6_SHOTS)
        print(f"\n  [V6] Depth sweep (natural decoherence): {len(V6_DEPTHS)} depths x {V6_SHOTS} shots")
        print(f"       Depths: {V6_DEPTHS}")
        print(f"       NO injected noise -- pure hardware vacuum")
        print(f"       Estimated cost: ${cost:.2f}")

        if not self._check_budget(cost, "V6"):
            return {'status': 'SKIPPED'}

        block_data = {
            'description': 'Depth sweep with natural QPU decoherence only',
            'circuit_type': 'deep_chain',
            'measurement_basis': 'Z',
            'shots_per_depth': V6_SHOTS,
            'depths': V6_DEPTHS,
            'noise_injection': 'NONE (natural QPU noise only)',
            'measurements': []
        }

        for idx, depth in enumerate(V6_DEPTHS):
            t0 = time.time()
            circuit = self._build_deep_chain(depth)
            bitstrings = self._run_z_basis(circuit, V6_SHOTS)
            dt = time.time() - t0

            entry = {
                'depth': depth,
                'n_cnot_gates': depth * (N_QUBITS - 1),
                'bitstrings': bitstrings,
                'counts': dict(Counter(bitstrings)),
                'n_shots': len(bitstrings),
                'timestamp': datetime.now().isoformat(),
                'elapsed_s': round(dt, 2),
            }
            block_data['measurements'].append(entry)

            n_unique = len(set(bitstrings))
            hamming_mean = np.mean([sum(int(b) for b in bs) for bs in bitstrings])
            print(f"       [{idx+1}/{len(V6_DEPTHS)}] depth={depth:2d}  "
                  f"gates={depth*(N_QUBITS-1):3d}  "
                  f"unique={n_unique:3d}  <H>={hamming_mean:.2f}  ({dt:.1f}s)")

        block_data['status'] = 'COMPLETE'
        return block_data

    # -------------------------------------------------------------------------
    # V7: Vacuum communication (information survival)
    # -------------------------------------------------------------------------

    def V7_vacuum_communication(self) -> Dict:
        """
        VACUUM COMMUNICATION TEST

        Question: Does gamma_c mark where encoded information is lost?

        Protocol:
        1. ENCODE: Prepare 6-qubit state encoding a known bit pattern
           (X gate on qubit i if message bit i = '1')
        2. ENTANGLE: Apply CNOT cascade (same as E3)
        3. DECOHERE: Apply noise at various gamma levels
        4. MEASURE: Z-basis measurement (many shots)
        5. DECODE: For each qubit, majority vote across shots
        6. SCORE: Compare decoded bits to original message

        Expected result:
        - gamma << gamma_c: high accuracy (information survives entanglement + noise)
        - gamma ~ gamma_c: rapid drop (information-to-physics transition)
        - gamma >> gamma_c: ~50% accuracy (random, information lost)

        Sigma-c on accuracy(gamma) should peak at gamma_c,
        confirming it as the information loss boundary.
        """
        n_tasks = len(V7_MESSAGES) * len(V7_GAMMAS)
        cost = self._estimate_cost(n_tasks, V7_SHOTS)
        print(f"\n  [V7] Vacuum communication: {len(V7_MESSAGES)} messages x {len(V7_GAMMAS)} gammas x {V7_SHOTS} shots")
        print(f"       Messages: {V7_MESSAGES}")
        print(f"       Estimated cost: ${cost:.2f}")

        if not self._check_budget(cost, "V7"):
            return {'status': 'SKIPPED'}

        block_data = {
            'description': 'Vacuum communication: information survival through decoherence',
            'circuit_type': 'entangled_chain_with_encoding',
            'measurement_basis': 'Z',
            'shots': V7_SHOTS,
            'messages': V7_MESSAGES,
            'gamma_values': V7_GAMMAS,
            'measurements': []
        }

        for msg in V7_MESSAGES:
            print(f"\n       Message: {msg}")

            for gamma in V7_GAMMAS:
                t0 = time.time()

                # BUILD: encode message, entangle, decohere
                circuit = Circuit()
                # Step 1: Encode message bits as initial state
                for i, bit in enumerate(msg):
                    if bit == '1':
                        circuit.x(i)

                # Step 2: Entangle via CNOT cascade (E3 structure)
                circuit.h(0)
                for i in range(N_QUBITS - 1):
                    circuit.cnot(i, i + 1)
                    if gamma > 0:
                        seed = int(gamma * 10000)
                        self._add_dephasing(circuit, [i, i + 1],
                                            gamma * DEPHASING_FACTOR, seed=seed + i * 100)
                        self._add_amplitude_damping(circuit, [i, i + 1],
                                                    gamma * DAMPING_FACTOR, seed=seed + i * 100 + 50)

                # Step 3: Measure
                bitstrings = self._run_z_basis(circuit, V7_SHOTS)
                dt = time.time() - t0

                # Step 4: Decode via majority vote per qubit
                decoded_bits = []
                qubit_probs = []
                for qi in range(N_QUBITS):
                    ones = sum(int(bs[qi]) for bs in bitstrings)
                    prob_one = ones / len(bitstrings)
                    qubit_probs.append(prob_one)
                    decoded_bits.append('1' if prob_one > 0.5 else '0')
                decoded = ''.join(decoded_bits)

                # Step 5: Score
                correct = sum(a == b for a, b in zip(msg, decoded))
                accuracy = correct / len(msg)

                entry = {
                    'message': msg,
                    'gamma': gamma,
                    'decoded': decoded,
                    'accuracy': accuracy,
                    'qubit_probabilities': qubit_probs,
                    'bitstrings': bitstrings,
                    'counts': dict(Counter(bitstrings)),
                    'n_shots': len(bitstrings),
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_s': round(dt, 2),
                }
                block_data['measurements'].append(entry)

                match = "OK" if accuracy >= 0.83 else "LOST" if accuracy <= 0.5 else "partial"
                print(f"         gamma={gamma:.3f}  {msg} -> {decoded}  "
                      f"acc={accuracy:.1%}  [{match}]  ({dt:.1f}s)")

        block_data['status'] = 'COMPLETE'
        return block_data

    # -------------------------------------------------------------------------
    # Save data
    # -------------------------------------------------------------------------

    def save_data(self, filename: str = None) -> str:
        if filename is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = str(OUTPUT_DIR / f"vacuum_telescope_{self.mode}_{timestamp}.json")

        self.data['metadata']['timestamp_end'] = datetime.now().isoformat()
        self.data['metadata']['total_tasks'] = self.task_count
        self.data['metadata']['total_shots'] = self.total_shots
        self.data['metadata']['total_cost_usd'] = round(self.spent_usd, 2)
        self.data['metadata']['total_cost_eur'] = round(self.spent_usd / EUR_TO_USD, 2)

        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                v = convert(obj)
                if v is not obj:
                    return v
                return super().default(obj)

        with open(filename, 'w') as f:
            json.dump(self.data, f, cls=NumpyEncoder, indent=1)

        size_mb = Path(filename).stat().st_size / (1024 * 1024)
        print(f"\n  Data saved: {filename} ({size_mb:.1f} MB)")
        return filename

    # -------------------------------------------------------------------------
    # Run all blocks
    # -------------------------------------------------------------------------

    def run_all(self) -> str:
        print("=" * 70)
        print("  VACUUM TELESCOPE v1.0")
        print("  Probing quantum vacuum structure via shot-level sigma-c")
        print("=" * 70)
        print(f"  Device: {self.mode}")
        print(f"  Budget: EUR {self.budget_eur:.2f}")
        print(f"  Qubits: {N_QUBITS}")
        print(f"  gamma_c prior: 0.6737 (from E3)")
        print()

        # Estimate total cost
        total_est = (
            self._estimate_cost(len(V1_GAMMAS), V1_SHOTS) +
            self._estimate_cost(len(V2_GAMMAS) * 2, V2_SHOTS) +
            self._estimate_cost(V3_N_RUNS, V3_SHOTS) +
            self._estimate_cost(len(V4_GAMMAS), V4_SHOTS) +
            self._estimate_cost(len(V5_GAMMAS), V5_SHOTS) +
            self._estimate_cost(len(V6_DEPTHS), V6_SHOTS) +
            self._estimate_cost(len(V7_MESSAGES) * len(V7_GAMMAS), V7_SHOTS)
        )
        print(f"  Total estimated cost: ${total_est:.2f} (EUR {total_est/EUR_TO_USD:.2f})")
        print()

        blocks = [
            ('V1_z_sweep', self.V1_gamma_sweep),
            ('V2_witness', self.V2_witness_calibration),
            ('V3_temporal', self.V3_temporal_structure),
            ('V4_control', self.V4_control),
            ('V5_ghz', self.V5_ghz_comparison),
            ('V6_depth', self.V6_depth_sweep),
            ('V7_communication', self.V7_vacuum_communication),
        ]

        for name, func in blocks:
            try:
                self.data['blocks'][name] = func()
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                traceback.print_exc()
                self.data['blocks'][name] = {'status': 'ERROR', 'error': str(e)}

            # Budget check
            if self.mode == 'qpu':
                budget_usd = self.budget_eur * EUR_TO_USD
                pct = self.spent_usd / budget_usd * 100
                print(f"       Budget: ${self.spent_usd:.2f}/${budget_usd:.2f} ({pct:.0f}%)")

        # Summary
        print("\n" + "=" * 70)
        print("  VACUUM TELESCOPE COMPLETE")
        print("=" * 70)
        completed = sum(1 for b in self.data['blocks'].values() if b.get('status') == 'COMPLETE')
        print(f"  Blocks completed: {completed}/7")
        print(f"  Total tasks: {self.task_count}")
        print(f"  Total shots: {self.total_shots:,}")
        if self.mode == 'qpu':
            print(f"  Total cost: ${self.spent_usd:.2f} (EUR {self.spent_usd/EUR_TO_USD:.2f})")

        filename = self.save_data()
        return filename


# =============================================================================
# POST-HOC ANALYSIS
# =============================================================================

class VacuumAnalysis:
    """
    Analyze raw data from VacuumTelescope.
    ALL analysis is computed from stored raw bitstrings.
    """

    def __init__(self, data_file: str):
        print(f"  Loading: {data_file}")
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        print(f"  Blocks: {list(self.data['blocks'].keys())}")
        self.results = {}

    # -------------------------------------------------------------------------
    # Bitstring statistics
    # -------------------------------------------------------------------------

    @staticmethod
    def hamming_weights(bitstrings: List[str]) -> np.ndarray:
        """Hamming weight (number of 1s) for each bitstring."""
        return np.array([sum(int(b) for b in bs) for bs in bitstrings])

    @staticmethod
    def bitstring_entropy(bitstrings: List[str]) -> float:
        """Shannon entropy of bitstring frequency distribution."""
        counts = Counter(bitstrings)
        total = len(bitstrings)
        probs = np.array([c / total for c in counts.values()])
        return float(-np.sum(probs * np.log2(probs + 1e-15)))

    @staticmethod
    def pairwise_zz(bitstrings: List[str], q1: int, q2: int) -> float:
        """<Z_q1 Z_q2> computed from Z-basis bitstrings."""
        parity_sum = 0
        for bs in bitstrings:
            s1 = 1 - 2 * int(bs[q1])  # 0->+1, 1->-1
            s2 = 1 - 2 * int(bs[q2])
            parity_sum += s1 * s2
        return parity_sum / len(bitstrings)

    # -------------------------------------------------------------------------
    # Sigma-c computation
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_sigma_c(sigmas: np.ndarray, observables: np.ndarray,
                        smoothing_window: int = 5) -> Tuple[np.ndarray, float, float]:
        """chi(sigma) = |dO/dsigma|, sigma_c = argmax chi, kappa = peak/median."""
        if not HAS_SCIPY:
            # Fallback: simple gradient
            chi = np.abs(np.gradient(observables, sigmas))
            peak_idx = np.argmax(chi)
            sigma_c = sigmas[peak_idx]
            baseline = np.median(chi)
            kappa = chi[peak_idx] / (baseline + 1e-15)
            return chi, float(sigma_c), float(kappa)

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

    # -------------------------------------------------------------------------
    # Analysis 1: Gamma-resolved bitstring statistics
    # -------------------------------------------------------------------------

    def analyze_v1_statistics(self) -> Dict:
        """Compute bitstring statistics across gamma sweep."""
        v1 = self.data['blocks'].get('V1_z_sweep')
        if not v1 or v1.get('status') != 'COMPLETE':
            return {'status': 'SKIPPED'}

        print("\n  --- V1 Analysis: Bitstring statistics vs gamma ---")

        gammas = []
        entropies = []
        mean_hammings = []
        std_hammings = []
        kurtosis_hammings = []
        n_unique_list = []

        for meas in v1['measurements']:
            gamma = meas['gamma']
            bs_list = meas['bitstrings']

            hw = self.hamming_weights(bs_list)
            entropy = self.bitstring_entropy(bs_list)
            n_unique = len(set(bs_list))

            gammas.append(gamma)
            entropies.append(entropy)
            mean_hammings.append(float(np.mean(hw)))
            std_hammings.append(float(np.std(hw)))
            kurtosis_hammings.append(float(stats.kurtosis(hw)) if HAS_SCIPY else 0.0)
            n_unique_list.append(n_unique)

        gammas = np.array(gammas)
        entropies = np.array(entropies)
        mean_hammings = np.array(mean_hammings)

        # Sigma-c on each observable
        chi_entropy, sc_entropy, kappa_entropy = self.compute_sigma_c(gammas, entropies)
        chi_hamming, sc_hamming, kappa_hamming = self.compute_sigma_c(gammas, mean_hammings)

        result = {
            'gammas': gammas.tolist(),
            'entropy': entropies.tolist(),
            'mean_hamming': mean_hammings,
            'std_hamming': std_hammings,
            'kurtosis_hamming': kurtosis_hammings,
            'n_unique': n_unique_list,
            'sigma_c_entropy': sc_entropy,
            'kappa_entropy': kappa_entropy,
            'sigma_c_hamming': sc_hamming,
            'kappa_hamming': kappa_hamming,
        }

        print(f"  Entropy:         sigma_c = {sc_entropy:.4f}  kappa = {kappa_entropy:.2f}")
        print(f"  Mean Hamming:    sigma_c = {sc_hamming:.4f}  kappa = {kappa_hamming:.2f}")

        self.results['v1_statistics'] = result
        return result

    # -------------------------------------------------------------------------
    # Analysis 2: Entanglement witness reconstruction
    # -------------------------------------------------------------------------

    def analyze_witness(self) -> Dict:
        """
        Reconstruct entanglement witness from V1 (ZZ) + V2 (XX, YY) data.
        W = (XX + YY)/2 - |ZZ|
        """
        v1 = self.data['blocks'].get('V1_z_sweep')
        v2 = self.data['blocks'].get('V2_witness')
        if not v1 or not v2:
            return {'status': 'SKIPPED'}
        if v1.get('status') != 'COMPLETE' or v2.get('status') != 'COMPLETE':
            return {'status': 'SKIPPED'}

        print("\n  --- Witness reconstruction: W = (XX+YY)/2 - |ZZ| ---")

        q1, q2 = V2_QUBIT_PAIR

        # Build gamma -> ZZ map from V1
        v1_zz = {}
        for meas in v1['measurements']:
            gamma = round(meas['gamma'], 6)
            zz = self.pairwise_zz(meas['bitstrings'], q1, q2)
            v1_zz[gamma] = zz

        # Reconstruct witness at V2 gamma values
        witness_gammas = []
        witness_values = []

        for meas in v2['measurements']:
            gamma = round(meas['gamma'], 6)

            # XX and YY from V2
            xx_exp = self._expectation_from_pair(meas['XX']['bitstrings'])
            yy_exp = self._expectation_from_pair(meas['YY']['bitstrings'])

            # ZZ from V1 (find closest gamma)
            closest_gamma = min(v1_zz.keys(), key=lambda g: abs(g - gamma))
            zz_exp = v1_zz[closest_gamma]

            witness = (xx_exp + yy_exp) / 2 - abs(zz_exp)

            witness_gammas.append(gamma)
            witness_values.append(witness)

            entangled = "ENTANGLED" if witness < 0 else "SEPARABLE"
            print(f"  gamma={gamma:.4f}  W={witness:+.4f}  "
                  f"XX={xx_exp:+.3f} YY={yy_exp:+.3f} ZZ={zz_exp:+.3f}  [{entangled}]")

        gammas = np.array(witness_gammas)
        witnesses = np.array(witness_values)

        chi, sigma_c, kappa = self.compute_sigma_c(gammas, witnesses)

        # Find zero crossing
        zero_crossing = None
        for i in range(len(witnesses) - 1):
            if witnesses[i] < 0 and witnesses[i + 1] >= 0:
                # Linear interpolation
                frac = -witnesses[i] / (witnesses[i + 1] - witnesses[i])
                zero_crossing = gammas[i] + frac * (gammas[i + 1] - gammas[i])
                break

        result = {
            'gammas': gammas.tolist(),
            'witnesses': witnesses.tolist(),
            'sigma_c': sigma_c,
            'kappa': kappa,
            'zero_crossing': float(zero_crossing) if zero_crossing else None,
            'chi': chi.tolist(),
        }

        print(f"\n  Witness sigma_c = {sigma_c:.4f}  kappa = {kappa:.2f}")
        if zero_crossing:
            print(f"  Zero crossing (entangled->separable): gamma = {zero_crossing:.4f}")
            print(f"  E3 reference: gamma_c = 0.6737")
            print(f"  Deviation: {abs(zero_crossing - 0.6737) / 0.6737 * 100:.1f}%")

        self.results['witness'] = result
        return result

    def _expectation_from_pair(self, bitstrings: List[str]) -> float:
        """Parity expectation from 2-qubit bitstrings."""
        total = len(bitstrings)
        parity_sum = 0
        for bs in bitstrings:
            b0, b1 = int(bs[0]), int(bs[1])
            parity_sum += 1 - 2 * (b0 ^ b1)
        return parity_sum / total

    # -------------------------------------------------------------------------
    # Analysis 3: Batch-size sigma-c (THE VACUUM TEST)
    # -------------------------------------------------------------------------

    def analyze_batch_sigma_c(self, gamma_target: float = 0.674) -> Dict:
        """
        THE KEY VACUUM STRUCTURE TEST.

        At fixed gamma (near gamma_c), split shots into batches of size K.
        Compute observable for each batch. Sweep K as scale parameter.
        chi(K) = |d<O>/dK|.

        QFT prediction: O(K) ~ const + noise/sqrt(K), so chi flat -> kappa ~ 1
        Vacuum structure: chi has a peak at critical K -> kappa >> 1
        """
        v1 = self.data['blocks'].get('V1_z_sweep')
        if not v1 or v1.get('status') != 'COMPLETE':
            return {'status': 'SKIPPED'}

        print(f"\n  --- Batch-size sigma-c (VACUUM TEST) at gamma ~ {gamma_target} ---")

        # Find closest gamma
        all_gammas = [m['gamma'] for m in v1['measurements']]
        idx = int(np.argmin([abs(g - gamma_target) for g in all_gammas]))
        actual_gamma = all_gammas[idx]
        bitstrings = v1['measurements'][idx]['bitstrings']
        n_total = len(bitstrings)

        print(f"  Using gamma = {actual_gamma:.4f} ({n_total} shots)")

        # Batch sizes to test
        batch_sizes = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600]
        batch_sizes = [k for k in batch_sizes if k <= n_total // 3]  # need >= 3 batches

        # Observables to test
        observables_to_test = {
            'hamming_mean': lambda bs_batch: np.mean(self.hamming_weights(bs_batch)),
            'hamming_var': lambda bs_batch: np.var(self.hamming_weights(bs_batch)),
            'entropy': lambda bs_batch: self.bitstring_entropy(bs_batch),
            'n_unique': lambda bs_batch: len(set(bs_batch)),
        }

        all_results = {}

        for obs_name, obs_func in observables_to_test.items():
            batch_ks = []
            batch_means = []
            batch_stds = []

            for K in batch_sizes:
                n_batches = n_total // K
                values = []
                for b in range(n_batches):
                    batch = bitstrings[b * K:(b + 1) * K]
                    values.append(obs_func(batch))
                batch_ks.append(K)
                batch_means.append(float(np.mean(values)))
                batch_stds.append(float(np.std(values)))

            batch_ks = np.array(batch_ks)
            batch_means = np.array(batch_means)
            batch_stds = np.array(batch_stds)

            # Sigma-c on batch means
            chi, sigma_c, kappa = self.compute_sigma_c(batch_ks, batch_means)

            # Also check variance scaling: QFT predicts var ~ 1/K
            # Fit log(var) vs log(K), expect slope = -1
            if len(batch_stds) > 3 and np.all(batch_stds > 0):
                log_k = np.log(batch_ks)
                log_var = np.log(batch_stds ** 2)
                slope, intercept = np.polyfit(log_k, log_var, 1)
                scaling_exponent = slope  # should be -1.0 for Gaussian
            else:
                scaling_exponent = None

            all_results[obs_name] = {
                'batch_sizes': batch_ks.tolist(),
                'batch_means': batch_means.tolist(),
                'batch_stds': batch_stds.tolist(),
                'chi': chi.tolist(),
                'sigma_c': sigma_c,
                'kappa': kappa,
                'variance_scaling_exponent': scaling_exponent,
            }

            qft_status = "QFT-CONSISTENT" if kappa < 2.0 else "** ANOMALOUS **"
            scaling_str = f"slope={scaling_exponent:.2f}" if scaling_exponent else "N/A"
            print(f"  {obs_name:15s}: sigma_c={sigma_c:6.0f}  kappa={kappa:.2f}  "
                  f"var~K^{scaling_str}  [{qft_status}]")

        self.results['batch_sigma_c'] = {
            'gamma': actual_gamma,
            'n_shots': n_total,
            'observables': all_results,
        }
        return self.results['batch_sigma_c']

    # -------------------------------------------------------------------------
    # Analysis 4: Temporal correlations
    # -------------------------------------------------------------------------

    def analyze_temporal(self) -> Dict:
        """Test for temporal correlations between sequential runs."""
        v3 = self.data['blocks'].get('V3_temporal')
        if not v3 or v3.get('status') != 'COMPLETE':
            return {'status': 'SKIPPED'}

        print(f"\n  --- Temporal correlation analysis ---")

        entropies = []
        mean_hammings = []
        n_uniques = []
        timestamps = []

        for meas in v3['measurements']:
            bs = meas['bitstrings']
            hw = self.hamming_weights(bs)
            entropies.append(self.bitstring_entropy(bs))
            mean_hammings.append(float(np.mean(hw)))
            n_uniques.append(len(set(bs)))
            timestamps.append(meas['timestamp'])

        entropies = np.array(entropies)
        mean_hammings = np.array(mean_hammings)

        # Autocorrelation
        def autocorr(x):
            x = x - np.mean(x)
            if np.std(x) < 1e-10:
                return 0.0
            return float(np.corrcoef(x[:-1], x[1:])[0, 1])

        ac_entropy = autocorr(entropies)
        ac_hamming = autocorr(mean_hammings)

        # Run-to-run variation (CV)
        cv_entropy = float(np.std(entropies) / (np.mean(entropies) + 1e-10))
        cv_hamming = float(np.std(mean_hammings) / (np.mean(mean_hammings) + 1e-10))

        result = {
            'gamma': v3['gamma'],
            'n_runs': len(entropies),
            'entropies': entropies.tolist(),
            'mean_hammings': mean_hammings.tolist(),
            'n_uniques': n_uniques,
            'autocorr_entropy': ac_entropy,
            'autocorr_hamming': ac_hamming,
            'cv_entropy': cv_entropy,
            'cv_hamming': cv_hamming,
            'timestamps': timestamps,
        }

        vacuum_memory = (abs(ac_entropy) > 0.5 or abs(ac_hamming) > 0.5)
        print(f"  Entropy:  mean={np.mean(entropies):.3f}  CV={cv_entropy:.3f}  AC={ac_entropy:+.3f}")
        print(f"  Hamming:  mean={np.mean(mean_hammings):.3f}  CV={cv_hamming:.3f}  AC={ac_hamming:+.3f}")
        print(f"  Vacuum memory: {'** DETECTED **' if vacuum_memory else 'none (as expected by QFT)'}")

        self.results['temporal'] = result
        return result

    # -------------------------------------------------------------------------
    # Analysis 5: Depth sweep (natural decoherence)
    # -------------------------------------------------------------------------

    def analyze_depth_sweep(self) -> Dict:
        """Sigma-c on circuit depth (natural QPU decoherence)."""
        v6 = self.data['blocks'].get('V6_depth')
        if not v6 or v6.get('status') != 'COMPLETE':
            return {'status': 'SKIPPED'}

        print(f"\n  --- Depth sweep analysis (natural decoherence) ---")

        depths = []
        entropies = []
        mean_hammings = []

        for meas in v6['measurements']:
            bs = meas['bitstrings']
            hw = self.hamming_weights(bs)
            depths.append(meas['depth'])
            entropies.append(self.bitstring_entropy(bs))
            mean_hammings.append(float(np.mean(hw)))

        depths = np.array(depths, dtype=float)
        entropies = np.array(entropies)
        mean_hammings = np.array(mean_hammings)

        chi_e, sc_e, kappa_e = self.compute_sigma_c(depths, entropies)
        chi_h, sc_h, kappa_h = self.compute_sigma_c(depths, mean_hammings)

        result = {
            'depths': depths.tolist(),
            'entropies': entropies.tolist(),
            'mean_hammings': mean_hammings.tolist(),
            'sigma_c_entropy': sc_e,
            'kappa_entropy': kappa_e,
            'sigma_c_hamming': sc_h,
            'kappa_hamming': kappa_h,
        }

        print(f"  Entropy:  sigma_c = depth {sc_e:.0f}  kappa = {kappa_e:.2f}")
        print(f"  Hamming:  sigma_c = depth {sc_h:.0f}  kappa = {kappa_h:.2f}")

        if kappa_e > 2.0 or kappa_h > 2.0:
            n_gates = int(max(sc_e, sc_h)) * (N_QUBITS - 1)
            print(f"  ** NATURAL DECOHERENCE THRESHOLD at ~{n_gates} CNOT gates **")
            print(f"  This is a real vacuum coupling signature!")

        self.results['depth_sweep'] = result
        return result

    # -------------------------------------------------------------------------
    # Analysis 6: Vacuum communication (information survival)
    # -------------------------------------------------------------------------

    def analyze_vacuum_communication(self) -> Dict:
        """
        Analyze information survival through decoherence.
        Key output: accuracy(gamma) curve and its sigma-c.
        """
        v7 = self.data['blocks'].get('V7_communication')
        if not v7 or v7.get('status') != 'COMPLETE':
            return {'status': 'SKIPPED'}

        print(f"\n  --- Vacuum communication analysis ---")

        messages = list(set(m['message'] for m in v7['measurements']))
        gammas = sorted(set(m['gamma'] for m in v7['measurements']))

        # Build accuracy matrix: messages x gammas
        accuracy_matrix = {}
        for msg in messages:
            accuracy_matrix[msg] = {}
            for meas in v7['measurements']:
                if meas['message'] == msg:
                    accuracy_matrix[msg][meas['gamma']] = meas['accuracy']

        # Mean accuracy across all messages at each gamma
        mean_accuracy = []
        for gamma in gammas:
            accs = [accuracy_matrix[msg].get(gamma, 0.5) for msg in messages]
            mean_accuracy.append(float(np.mean(accs)))

        gammas_arr = np.array(gammas)
        accuracy_arr = np.array(mean_accuracy)

        # Sigma-c on accuracy curve
        chi, sigma_c, kappa = self.compute_sigma_c(gammas_arr, accuracy_arr)

        # Find 75% accuracy threshold (information boundary)
        info_boundary = None
        for i in range(len(accuracy_arr) - 1):
            if accuracy_arr[i] > 0.75 and accuracy_arr[i + 1] <= 0.75:
                frac = (accuracy_arr[i] - 0.75) / (accuracy_arr[i] - accuracy_arr[i + 1])
                info_boundary = gammas_arr[i] + frac * (gammas_arr[i + 1] - gammas_arr[i])
                break

        # Per-message results
        print(f"\n  {'Message':>8s}", end="")
        for gamma in gammas:
            print(f"  {gamma:.2f}", end="")
        print()
        for msg in messages:
            print(f"  {msg}", end="")
            for gamma in gammas:
                acc = accuracy_matrix[msg].get(gamma, 0.5)
                marker = "+" if acc > 0.75 else "-" if acc < 0.55 else "~"
                print(f"  {acc:.0%}{marker}", end="")
            print()

        print(f"\n  Mean accuracy curve:")
        for gamma, acc in zip(gammas, mean_accuracy):
            bar = "#" * int(acc * 20)
            print(f"    gamma={gamma:.3f}  acc={acc:.1%}  {bar}")

        print(f"\n  Sigma-c (max information loss rate): gamma = {sigma_c:.4f}  kappa = {kappa:.2f}")
        if info_boundary:
            print(f"  75% accuracy boundary: gamma = {info_boundary:.4f}")
            print(f"  E3 reference gamma_c:  0.6737")
            dev = abs(info_boundary - 0.6737) / 0.6737 * 100
            if dev < 10:
                print(f"  -> CONSISTENT: information boundary matches entanglement boundary ({dev:.1f}%)")
            else:
                print(f"  -> Deviation: {dev:.1f}% (different thresholds for different information types)")

        result = {
            'messages': messages,
            'gammas': gammas,
            'accuracy_matrix': {msg: list(accuracy_matrix[msg].values()) for msg in messages},
            'mean_accuracy': mean_accuracy,
            'sigma_c': sigma_c,
            'kappa': kappa,
            'info_boundary_75pct': float(info_boundary) if info_boundary else None,
            'chi': chi.tolist(),
        }

        self.results['vacuum_communication'] = result
        return result

    # -------------------------------------------------------------------------
    # Analysis 7: Cross-circuit comparison
    # -------------------------------------------------------------------------

    def analyze_cross_circuit(self) -> Dict:
        """Compare entangled chain, GHZ, and product state at similar gammas."""
        print(f"\n  --- Cross-circuit comparison ---")

        comparison = {}

        # Get data from each block
        for block_name, gamma_key, label in [
            ('V1_z_sweep', 'gamma', 'chain'),
            ('V5_ghz', 'gamma', 'ghz'),
            ('V4_control', 'gamma', 'product'),
        ]:
            block = self.data['blocks'].get(block_name)
            if not block or block.get('status') != 'COMPLETE':
                continue

            for meas in block['measurements']:
                gamma = round(meas[gamma_key], 2)
                bs = meas['bitstrings']
                hw = self.hamming_weights(bs)

                key = f"{label}_g{gamma}"
                comparison[key] = {
                    'circuit': label,
                    'gamma': gamma,
                    'entropy': self.bitstring_entropy(bs),
                    'mean_hamming': float(np.mean(hw)),
                    'std_hamming': float(np.std(hw)),
                    'n_unique': len(set(bs)),
                }

        # Print comparison table
        print(f"  {'Circuit':10s} {'gamma':>6s} {'Entropy':>8s} {'<H>':>6s} {'std(H)':>7s} {'Unique':>7s}")
        print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*7}")
        for key in sorted(comparison.keys()):
            c = comparison[key]
            print(f"  {c['circuit']:10s} {c['gamma']:6.2f} {c['entropy']:8.3f} "
                  f"{c['mean_hamming']:6.2f} {c['std_hamming']:7.3f} {c['n_unique']:7d}")

        self.results['cross_circuit'] = comparison
        return comparison

    # -------------------------------------------------------------------------
    # Full analysis
    # -------------------------------------------------------------------------

    def full_analysis(self):
        """Run all analyses and print summary."""
        print("=" * 70)
        print("  VACUUM TELESCOPE ANALYSIS")
        print("=" * 70)

        self.analyze_v1_statistics()
        self.analyze_witness()
        self.analyze_batch_sigma_c(gamma_target=0.674)
        self.analyze_batch_sigma_c(gamma_target=0.0)    # control: far from transition
        self.analyze_temporal()
        self.analyze_depth_sweep()
        self.analyze_vacuum_communication()
        self.analyze_cross_circuit()

        # Final verdict
        print("\n" + "=" * 70)
        print("  VERDICT")
        print("=" * 70)

        # Check batch sigma-c at gamma_c
        batch = self.results.get('batch_sigma_c', {})
        if batch:
            anomalous = []
            for obs_name, obs_result in batch.get('observables', {}).items():
                if obs_result.get('kappa', 0) > 2.0:
                    anomalous.append(f"{obs_name} (kappa={obs_result['kappa']:.2f})")

            if anomalous:
                print(f"\n  ** POTENTIAL VACUUM STRUCTURE DETECTED **")
                print(f"  Anomalous batch-size scaling in:")
                for a in anomalous:
                    print(f"    - {a}")
                print(f"  This requires further investigation:")
                print(f"    1. Verify not a statistical fluke (run V3 many more times)")
                print(f"    2. Check if product state (V4) shows same effect")
                print(f"    3. Compare with GHZ (V5) at same gamma")
            else:
                print(f"\n  Batch-size sigma-c: kappa ~ 1 for all observables")
                print(f"  -> Shot statistics are scale-invariant (QFT-consistent)")
                print(f"  -> No vacuum structure detected at this resolution")

        # Check vacuum communication
        vcomm = self.results.get('vacuum_communication', {})
        if vcomm and vcomm.get('info_boundary_75pct'):
            ib = vcomm['info_boundary_75pct']
            print(f"\n  Vacuum communication: information boundary at gamma = {ib:.4f}")
            if vcomm.get('kappa', 0) > 2.0:
                print(f"  ** Sharp information loss (kappa = {vcomm['kappa']:.2f}) **")
                print(f"  -> Decoherence creates a REAL information boundary")

        # Check witness
        wit = self.results.get('witness', {})
        if wit and wit.get('zero_crossing'):
            zc = wit['zero_crossing']
            deviation = abs(zc - 0.6737) / 0.6737 * 100
            print(f"\n  Witness zero-crossing: gamma = {zc:.4f} (E3 ref: 0.6737, dev: {deviation:.1f}%)")
            if deviation < 5:
                print(f"  -> Confirms E3 result: transition at gamma_c ~ 0.67")
            else:
                print(f"  -> Deviation from E3. Higher resolution may be needed.")

        # Check depth sweep
        depth = self.results.get('depth_sweep', {})
        if depth and max(depth.get('kappa_entropy', 0), depth.get('kappa_hamming', 0)) > 2.0:
            print(f"\n  ** NATURAL DECOHERENCE THRESHOLD DETECTED **")
            print(f"  The real QPU vacuum has a critical circuit depth.")
            print(f"  This is an operational vacuum coupling constant.")

        print()

        # Save analysis results
        output_file = str(Path(self.data['metadata'].get('timestamp_start', 'analysis')
                               ).parent / "vacuum_analysis.json")
        # Use a simpler path
        output_file = "data/vacuum_telescope_v1/vacuum_analysis.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=lambda x: x.tolist()
                      if isinstance(x, np.ndarray) else float(x)
                      if isinstance(x, (np.floating, np.integer)) else x)
        print(f"  Analysis saved: {output_file}")

        return self.results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vacuum Telescope v1.0")
    parser.add_argument('--mode', choices=['sim', 'qpu'], default='sim',
                        help='sim = local simulator, qpu = Rigetti Ankaa-3')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Path to data file for post-hoc analysis')
    parser.add_argument('--budget', type=float, default=BUDGET_EUR,
                        help=f'Budget in EUR (default: {BUDGET_EUR})')
    args = parser.parse_args()

    if args.analyze:
        # Analysis mode
        analyzer = VacuumAnalysis(args.analyze)
        analyzer.full_analysis()
    else:
        # Experiment mode
        telescope = VacuumTelescope(mode=args.mode, budget_eur=args.budget)
        filename = telescope.run_all()
        print(f"\n  Now run analysis with:")
        print(f"  python vacuum_telescope.py --analyze {filename}")


if __name__ == "__main__":
    main()
