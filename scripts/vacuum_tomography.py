#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  VACUUM TOMOGRAPHY v1.0                                                      ║
║  "Focusing on the choreography"                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import signal, stats, ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.fft import fft, ifft, fftfreq
import h5py
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from itertools import permutations
from datetime import datetime
import traceback

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path("data_pool")
OUTPUT_DIR = Path("vacuum_tomography_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 64  # Hz

# Time windows
WINDOWS = {
    'PRE':       (-300, -60),
    'PRECURSOR': (-60, -5),
    'EVENT':     (-5, +60),
    'POST':      (+60, +300),
}

@dataclass
class EventConfig:
    name: str
    gps_time: int
    magnitude: float
    location: str
    lat: float
    lon: float
    depth_km: float
    h1_file: str
    l1_file: str

# Events from data_pool
EVENTS = [
    EventConfig("Peru_M8.0", 1243310085, 8.0, "Peru", -5.81, -75.27, 122, 
                "H-H1_GWOSC_O3a_4KHZ_R1-1243308032-4096.hdf5", "L-L1_GWOSC_O3a_4KHZ_R1-1243308032-4096.hdf5"),
    EventConfig("Jamaica_M7.7", 1264403810, 7.7, "Jamaica", 19.42, -78.76, 15, 
                "H-H1_GWOSC_O3b_4KHZ_R1-1264402432-4096.hdf5", "L-L1_GWOSC_O3b_4KHZ_R1-1264402432-4096.hdf5"),
    EventConfig("PNG_M7.6", 1242697098, 7.6, "Papua New Guinea", -4.05, 152.55, 10, 
                "H-H1_GWOSC_O3a_4KHZ_R1-1242693632-4096.hdf5", "L-L1_GWOSC_O3a_4KHZ_R1-1242693632-4096.hdf5"),
    EventConfig("PNG_M7.1", 1241800887, 7.1, "Papua New Guinea", -5.58, 151.2, 48, 
                "H-H1_GWOSC_O3a_4KHZ_R1-1241800704-4096.hdf5", "L-L1_GWOSC_O3a_4KHZ_R1-1241800704-4096.hdf5"),
    EventConfig("Indonesia_M6.8", 1248623540, 6.8, "Indonesia", 0.51, 122.38, 30, 
                "H-H1_GWOSC_O3a_4KHZ_R1-1248620544-4096.hdf5", "L-L1_GWOSC_O3a_4KHZ_R1-1248620544-4096.hdf5"),
    EventConfig("Turkey_M6.8", 1264070540, 6.8, "Turkey", 38.39, 39.1, 10, 
                "H-H1_GWOSC_O3b_4KHZ_R1-1264066560-4096.hdf5", "L-L1_GWOSC_O3b_4KHZ_R1-1264066560-4096.hdf5"),
    EventConfig("Chile_M6.7", 1248971282, 6.7, "Chile", -32.09, -71.48, 53, 
                "H-H1_GWOSC_O3a_4KHZ_R1-1248968704-4096.hdf5", "L-L1_GWOSC_O3a_4KHZ_R1-1248968704-4096.hdf5"),
    EventConfig("SandwichIs_M6.6a", 1252013140, 6.6, "South Sandwich Islands", -57.71, -25.27, 101, 
                "H-H1_GWOSC_O3a_4KHZ_R1-1252012032-4096.hdf5", "L-L1_GWOSC_O3a_4KHZ_R1-1252012032-4096.hdf5"),
]

# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

def load_ligo_data(event: EventConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Load and downsample LIGO data."""
    h1_path = DATA_DIR / event.h1_file
    l1_path = DATA_DIR / event.l1_file
    
    if not h1_path.exists() or not l1_path.exists():
        return None, None, 0
    
    try:
        with h5py.File(h1_path, 'r') as f:
            h1_raw = f['strain/Strain'][:]
            h1_t0 = f['strain/Strain'].attrs['Xstart']
        
        with h5py.File(l1_path, 'r') as f:
            l1_raw = f['strain/Strain'][:]
        
        factor = 4096 // SAMPLE_RATE
        h1 = signal.decimate(h1_raw, factor, zero_phase=True)
        l1 = signal.decimate(l1_raw, factor, zero_phase=True)
        
        eq_offset = int((event.gps_time - h1_t0) * SAMPLE_RATE)
        
        return h1, l1, eq_offset
        
    except Exception as e:
        print(f"    ⚠️ Error: {e}")
        return None, None, 0

def extract_window(data: np.ndarray, eq_offset: int, t_start: int, t_end: int) -> Optional[np.ndarray]:
    """Extract and normalize a time window."""
    start = int(eq_offset + t_start * SAMPLE_RATE)
    end = int(eq_offset + t_end * SAMPLE_RATE)
    
    start = max(0, start)
    end = min(len(data), end)
    
    if end - start < SAMPLE_RATE * 5:
        return None
    
    window = data[start:end].copy()
    window = (window - np.mean(window)) / (np.std(window) + 1e-12)
    
    return window

# ════════════════════════════════════════════════════════════════════════════════
# METHOD 1: SYMBOLIC DYNAMICS
# ════════════════════════════════════════════════════════════════════════════════

def signal_to_symbols(x: np.ndarray, n_symbols: int = 4) -> np.ndarray:
    """Convert signal to symbolic sequence using quantiles."""
    quantiles = np.percentile(x, np.linspace(0, 100, n_symbols + 1))
    symbols = np.digitize(x, quantiles[1:-1])
    return symbols

def compute_word_distribution(symbols: np.ndarray, word_length: int = 3) -> Dict[str, float]:
    """Compute distribution of symbolic words."""
    words = []
    for i in range(len(symbols) - word_length + 1):
        word = ''.join(map(str, symbols[i:i+word_length]))
        words.append(word)
    
    counts = Counter(words)
    total = sum(counts.values())
    
    return {w: c/total for w, c in counts.items()}

def symbolic_dynamics_analysis(x: np.ndarray, n_symbols: int = 4, word_length: int = 3) -> Dict:
    """Complete symbolic dynamics analysis."""
    symbols = signal_to_symbols(x, n_symbols)
    word_dist = compute_word_distribution(symbols, word_length)
    
    # Entropy of word distribution
    probs = np.array(list(word_dist.values()))
    probs = probs[probs > 0]
    word_entropy = -np.sum(probs * np.log2(probs))
    
    # Maximum possible entropy
    max_entropy = word_length * np.log2(n_symbols)
    
    # Forbidden words (exist in theory but not in data)
    all_possible = n_symbols ** word_length
    n_observed = len(word_dist)
    n_forbidden = all_possible - n_observed
    
    # Most common patterns
    top_words = sorted(word_dist.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'word_entropy': float(word_entropy),
        'max_entropy': float(max_entropy),
        'entropy_ratio': float(word_entropy / max_entropy),
        'n_observed_words': n_observed,
        'n_forbidden_words': n_forbidden,
        'forbidden_ratio': float(n_forbidden / all_possible),
        'top_words': top_words,
        'word_distribution': word_dist,
    }

def compare_symbolic_grammars(pre: np.ndarray, post: np.ndarray) -> Dict:
    """Compare the symbolic grammar between two windows."""
    pre_analysis = symbolic_dynamics_analysis(pre)
    post_analysis = symbolic_dynamics_analysis(post)
    
    # Words that appear in POST but not in PRE (new patterns!)
    pre_words = set(pre_analysis['word_distribution'].keys())
    post_words = set(post_analysis['word_distribution'].keys())
    
    new_words = post_words - pre_words
    lost_words = pre_words - post_words
    
    # KL divergence between distributions
    common_words = pre_words & post_words
    if common_words:
        kl_div = 0
        for w in common_words:
            p = pre_analysis['word_distribution'][w]
            q = post_analysis['word_distribution'].get(w, 1e-10)
            kl_div += p * np.log2(p / q)
    else:
        kl_div = float('inf')
    
    return {
        'pre': pre_analysis,
        'post': post_analysis,
        'new_patterns': list(new_words)[:10],
        'lost_patterns': list(lost_words)[:10],
        'n_new_patterns': len(new_words),
        'n_lost_patterns': len(lost_words),
        'grammar_change': len(new_words) + len(lost_words),
        'kl_divergence': float(kl_div) if kl_div != float('inf') else 999.0,
        'entropy_change': post_analysis['word_entropy'] - pre_analysis['word_entropy'],
    }

# ════════════════════════════════════════════════════════════════════════════════
# METHOD 2: RECURRENCE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def compute_recurrence_matrix(x: np.ndarray, embedding_dim: int = 3, 
                               delay: int = 1, threshold: float = 0.1) -> np.ndarray:
    """Compute recurrence matrix using time-delay embedding."""
    # Create embedded vectors
    n = len(x) - (embedding_dim - 1) * delay
    embedded = np.zeros((n, embedding_dim))
    
    for i in range(embedding_dim):
        embedded[:, i] = x[i*delay:i*delay + n]
    
    # Compute distance matrix
    distances = squareform(pdist(embedded, 'euclidean'))
    
    # Threshold to get recurrence matrix
    threshold_val = threshold * np.std(distances)
    recurrence = (distances < threshold_val).astype(int)
    
    return recurrence

def recurrence_quantification(R: np.ndarray) -> Dict:
    """Compute Recurrence Quantification Analysis (RQA) measures."""
    n = R.shape[0]
    
    # Recurrence Rate (RR)
    rr = np.sum(R) / (n * n)
    
    # Determinism (DET) - ratio of recurrence points in diagonal lines
    diag_lines = []
    for k in range(-n+1, n):
        diag = np.diag(R, k)
        # Find line lengths
        line_lengths = []
        current_length = 0
        for val in diag:
            if val == 1:
                current_length += 1
            else:
                if current_length >= 2:
                    line_lengths.append(current_length)
                current_length = 0
        if current_length >= 2:
            line_lengths.append(current_length)
        diag_lines.extend(line_lengths)
    
    if diag_lines:
        det = sum(diag_lines) / (np.sum(R) + 1e-10)
        avg_diag_length = np.mean(diag_lines)
        max_diag_length = max(diag_lines)
        diag_entropy = stats.entropy(np.bincount(diag_lines)[1:] + 1e-10)
    else:
        det = 0
        avg_diag_length = 0
        max_diag_length = 0
        diag_entropy = 0
    
    # Laminarity (LAM) - ratio of recurrence points in vertical lines
    vert_lines = []
    for col in range(n):
        column = R[:, col]
        line_lengths = []
        current_length = 0
        for val in column:
            if val == 1:
                current_length += 1
            else:
                if current_length >= 2:
                    line_lengths.append(current_length)
                current_length = 0
        if current_length >= 2:
            line_lengths.append(current_length)
        vert_lines.extend(line_lengths)
    
    if vert_lines:
        lam = sum(vert_lines) / (np.sum(R) + 1e-10)
        avg_vert_length = np.mean(vert_lines)
    else:
        lam = 0
        avg_vert_length = 0
    
    return {
        'recurrence_rate': float(rr),
        'determinism': float(det),
        'laminarity': float(lam),
        'avg_diagonal_length': float(avg_diag_length),
        'max_diagonal_length': int(max_diag_length),
        'diagonal_entropy': float(diag_entropy),
        'avg_vertical_length': float(avg_vert_length),
    }

def recurrence_analysis(x: np.ndarray, subsample: int = 4) -> Dict:
    """Full recurrence analysis with subsampling for speed."""
    x_sub = x[::subsample]
    R = compute_recurrence_matrix(x_sub, embedding_dim=3, delay=2, threshold=0.15)
    rqa = recurrence_quantification(R)
    return rqa

# ════════════════════════════════════════════════════════════════════════════════
# METHOD 3: ORDINAL PATTERNS (Permutation Entropy)
# ════════════════════════════════════════════════════════════════════════════════

def ordinal_patterns(x: np.ndarray, order: int = 3, delay: int = 1) -> Dict:
    """Compute ordinal patterns and permutation entropy."""
    n = len(x) - (order - 1) * delay
    
    # All possible permutations
    perms = list(permutations(range(order)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Count pattern occurrences
    pattern_counts = np.zeros(len(perms))
    
    for i in range(n):
        # Extract embedded vector
        vec = [x[i + j*delay] for j in range(order)]
        # Get ordinal pattern (rank ordering)
        pattern = tuple(np.argsort(vec))
        pattern_counts[perm_to_idx[pattern]] += 1
    
    # Normalize to probabilities
    probs = pattern_counts / n
    probs = probs[probs > 0]
    
    # Permutation entropy
    perm_entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(perms))
    
    # Statistical complexity (Jensen-Shannon complexity)
    uniform = np.ones(len(perms)) / len(perms)
    full_probs = pattern_counts / n
    
    # JS divergence
    m = (full_probs + uniform) / 2
    js_div = 0.5 * np.sum(full_probs * np.log2(full_probs / m + 1e-10)) + \
             0.5 * np.sum(uniform * np.log2(uniform / m + 1e-10))
    
    complexity = (perm_entropy / max_entropy) * js_div
    
    return {
        'permutation_entropy': float(perm_entropy),
        'normalized_pe': float(perm_entropy / max_entropy),
        'statistical_complexity': float(complexity),
        'n_patterns_observed': int(np.sum(pattern_counts > 0)),
        'n_patterns_possible': len(perms),
        'dominant_pattern': perms[np.argmax(pattern_counts)],
        'pattern_distribution': pattern_counts.tolist(),
    }

# ════════════════════════════════════════════════════════════════════════════════
# METHOD 4: TRANSFER ENTROPY (Information Flow)
# ════════════════════════════════════════════════════════════════════════════════

def transfer_entropy(source: np.ndarray, target: np.ndarray, 
                     k: int = 1, l: int = 1, delay: int = 1, n_bins: int = 8) -> float:
    """
    Compute transfer entropy from source to target.
    TE(X→Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)
    
    High TE(X→Y) means X "causes" or "leads" Y.
    """
    n = len(source) - max(k, l) * delay
    
    # Discretize
    source_binned = np.digitize(source, np.linspace(source.min(), source.max(), n_bins))
    target_binned = np.digitize(target, np.linspace(target.min(), target.max(), n_bins))
    
    # Build joint distributions
    # Y_t, Y_past, X_past
    joint_counts = {}
    marginal_y_ypast = {}
    marginal_ypast_xpast = {}
    marginal_ypast = {}
    
    for i in range(n):
        y_t = target_binned[i + k*delay]
        y_past = tuple(target_binned[i + j*delay] for j in range(k))
        x_past = tuple(source_binned[i + j*delay] for j in range(l))
        
        key_full = (y_t, y_past, x_past)
        key_y_ypast = (y_t, y_past)
        key_ypast_xpast = (y_past, x_past)
        key_ypast = y_past
        
        joint_counts[key_full] = joint_counts.get(key_full, 0) + 1
        marginal_y_ypast[key_y_ypast] = marginal_y_ypast.get(key_y_ypast, 0) + 1
        marginal_ypast_xpast[key_ypast_xpast] = marginal_ypast_xpast.get(key_ypast_xpast, 0) + 1
        marginal_ypast[key_ypast] = marginal_ypast.get(key_ypast, 0) + 1
    
    # Compute transfer entropy
    te = 0
    for (y_t, y_past, x_past), count in joint_counts.items():
        p_joint = count / n
        p_y_ypast = marginal_y_ypast[(y_t, y_past)] / n
        p_ypast_xpast = marginal_ypast_xpast[(y_past, x_past)] / n
        p_ypast = marginal_ypast[y_past] / n
        
        # TE = sum p(y_t, y_past, x_past) * log(p(y_t | y_past, x_past) / p(y_t | y_past))
        if p_ypast_xpast > 0 and p_ypast > 0:
            te += p_joint * np.log2((p_joint * p_ypast) / (p_y_ypast * p_ypast_xpast + 1e-10) + 1e-10)
    
    return max(0, te)

def bidirectional_information_flow(h1: np.ndarray, l1: np.ndarray) -> Dict:
    """Compute information flow between H1 and L1."""
    te_h1_to_l1 = transfer_entropy(h1, l1, k=2, l=2, delay=1)
    te_l1_to_h1 = transfer_entropy(l1, h1, k=2, l=2, delay=1)
    
    # Net information flow
    net_flow = te_h1_to_l1 - te_l1_to_h1
    
    # Asymmetry index
    total = te_h1_to_l1 + te_l1_to_h1
    asymmetry = net_flow / (total + 1e-10)
    
    return {
        'te_h1_to_l1': float(te_h1_to_l1),
        'te_l1_to_h1': float(te_l1_to_h1),
        'net_flow': float(net_flow),
        'asymmetry_index': float(asymmetry),
        'dominant_direction': 'H1→L1' if net_flow > 0 else 'L1→H1',
        'total_information_flow': float(total),
    }

# ════════════════════════════════════════════════════════════════════════════════
# METHOD 5: PHASE SPACE TOPOLOGY
# ════════════════════════════════════════════════════════════════════════════════

def phase_space_analysis(x: np.ndarray, embedding_dim: int = 3, delay: int = 2) -> Dict:
    """Analyze the topology of the phase space attractor."""
    n = len(x) - (embedding_dim - 1) * delay
    
    # Create embedded vectors
    embedded = np.zeros((n, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = x[i*delay:i*delay + n]
    
    # Correlation dimension estimate (simplified)
    distances = pdist(embedded[::10], 'euclidean')  # Subsample for speed
    distances = distances[distances > 0]
    
    # Count pairs within different radii
    radii = np.percentile(distances, [10, 20, 30, 40, 50])
    counts = [np.sum(distances < r) for r in radii]
    
    # Estimate correlation dimension from slope
    if len(counts) > 1 and all(c > 0 for c in counts):
        log_r = np.log(radii)
        log_c = np.log(counts)
        corr_dim, _ = np.polyfit(log_r, log_c, 1)
    else:
        corr_dim = 0
    
    # Attractor size metrics
    attractor_volume = np.prod(np.std(embedded, axis=0))
    attractor_extent = np.max(embedded, axis=0) - np.min(embedded, axis=0)
    
    # Density estimation at center
    center = np.mean(embedded, axis=0)
    distances_to_center = np.linalg.norm(embedded - center, axis=1)
    central_density = np.sum(distances_to_center < np.percentile(distances_to_center, 20)) / n
    
    return {
        'correlation_dimension': float(corr_dim),
        'attractor_volume': float(attractor_volume),
        'attractor_extent': attractor_extent.tolist(),
        'central_density': float(central_density),
        'mean_position': center.tolist(),
    }

# ════════════════════════════════════════════════════════════════════════════════
# METHOD 6: VISIBILITY GRAPH
# ════════════════════════════════════════════════════════════════════════════════

def visibility_graph_metrics(x: np.ndarray, max_points: int = 500) -> Dict:
    """
    Convert time series to visibility graph and compute network metrics.
    Two points are connected if no intermediate point blocks the "line of sight".
    """
    # Subsample for computational feasibility
    if len(x) > max_points:
        indices = np.linspace(0, len(x)-1, max_points, dtype=int)
        x = x[indices]
    
    n = len(x)
    
    # Build adjacency matrix using natural visibility
    adjacency = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i+2, min(i+50, n)):  # Limit range for speed
            # Check if all intermediate points are below the line connecting i and j
            visible = True
            for k in range(i+1, j):
                # Line equation: y = y_i + (y_j - y_i) * (k - i) / (j - i)
                line_height = x[i] + (x[j] - x[i]) * (k - i) / (j - i)
                if x[k] >= line_height:
                    visible = False
                    break
            
            if visible:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
    
    # Network metrics
    degrees = np.sum(adjacency, axis=1)
    
    # Degree distribution
    mean_degree = np.mean(degrees)
    degree_std = np.std(degrees)
    max_degree = np.max(degrees)
    
    # Clustering coefficient (simplified - local)
    clustering_coeffs = []
    for i in range(n):
        neighbors = np.where(adjacency[i] == 1)[0]
        k = len(neighbors)
        if k >= 2:
            # Count edges between neighbors
            edges_between = 0
            for ni in range(len(neighbors)):
                for nj in range(ni+1, len(neighbors)):
                    if adjacency[neighbors[ni], neighbors[nj]] == 1:
                        edges_between += 1
            clustering_coeffs.append(2 * edges_between / (k * (k-1)))
        else:
            clustering_coeffs.append(0)
    
    avg_clustering = np.mean(clustering_coeffs)
    
    # Assortativity (degree correlation)
    edges = np.array(np.where(adjacency == 1)).T
    if len(edges) > 0:
        degree_i = degrees[edges[:, 0]]
        degree_j = degrees[edges[:, 1]]
        assortativity = np.corrcoef(degree_i, degree_j)[0, 1]
    else:
        assortativity = 0
    
    return {
        'mean_degree': float(mean_degree),
        'degree_std': float(degree_std),
        'max_degree': int(max_degree),
        'avg_clustering': float(avg_clustering),
        'assortativity': float(assortativity) if not np.isnan(assortativity) else 0,
        'n_edges': int(np.sum(adjacency) // 2),
        'edge_density': float(np.sum(adjacency) / (n * (n-1))),
    }

# ════════════════════════════════════════════════════════════════════════════════
# METHOD 7: MULTI-SCALE TIME-FREQUENCY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def morlet_wavelet(t: np.ndarray, scale: float, omega0: float = 6.0) -> np.ndarray:
    """Generate Morlet wavelet."""
    scaled_t = t / scale
    return np.exp(1j * omega0 * scaled_t) * np.exp(-0.5 * scaled_t**2) / np.sqrt(scale)

def manual_cwt(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Manual continuous wavelet transform using convolution."""
    n = len(x)
    coefficients = np.zeros((len(scales), n), dtype=complex)
    
    for i, scale in enumerate(scales):
        # Create wavelet
        wavelet_length = min(int(10 * scale), n)
        t = np.arange(-wavelet_length//2, wavelet_length//2)
        wavelet = morlet_wavelet(t, scale)
        
        # Convolve
        conv = np.convolve(x, wavelet, mode='same')
        coefficients[i, :] = conv
    
    return coefficients

def wavelet_analysis(x: np.ndarray, scales: np.ndarray = None) -> Dict:
    """Multi-scale time-frequency analysis."""
    if scales is None:
        scales = np.array([2, 4, 8, 16, 32, 64])
        scales = scales[scales < len(x)//4]
    
    if len(scales) == 0:
        scales = np.array([2, 4, 8])
    
    # Manual CWT
    coefficients = manual_cwt(x, scales)
    
    # Power
    power = np.abs(coefficients) ** 2
    
    # Mean power at each scale
    mean_power = np.mean(power, axis=1)
    
    # Dominant scale
    dominant_scale = scales[np.argmax(mean_power)]
    
    # Scale entropy (how spread is the power across scales)
    power_dist = mean_power / (np.sum(mean_power) + 1e-10)
    scale_entropy = -np.sum(power_dist * np.log2(power_dist + 1e-10))
    
    # Time evolution of dominant frequency
    dominant_scale_over_time = scales[np.argmax(power, axis=0)]
    scale_variability = np.std(dominant_scale_over_time)
    
    return {
        'dominant_scale': int(dominant_scale),
        'scale_entropy': float(scale_entropy),
        'scale_variability': float(scale_variability),
        'total_power': float(np.sum(power)),
        'power_by_scale': mean_power.tolist(),
    }

def wavelet_coherence(h1: np.ndarray, l1: np.ndarray) -> Dict:
    """Compute wavelet coherence between H1 and L1."""
    scales = np.array([2, 4, 8, 16, 32])
    scales = scales[scales < len(h1)//8]
    
    if len(scales) == 0:
        scales = np.array([2, 4])
    
    # CWT of both signals
    cwt_h1 = manual_cwt(h1, scales)
    cwt_l1 = manual_cwt(l1, scales)
    
    # Cross-wavelet spectrum
    cross_spectrum = cwt_h1 * np.conj(cwt_l1)
    
    # Wavelet coherence (smoothed)
    smooth_window = 5
    
    power_h1 = ndimage.uniform_filter(np.abs(cwt_h1)**2, size=smooth_window)
    power_l1 = ndimage.uniform_filter(np.abs(cwt_l1)**2, size=smooth_window)
    cross_smooth = ndimage.uniform_filter(np.abs(cross_spectrum), size=smooth_window)
    
    coherence = cross_smooth**2 / (power_h1 * power_l1 + 1e-10)
    
    # Mean coherence at each scale
    mean_coherence = np.mean(np.real(coherence), axis=1)
    
    # Phase difference
    phase_diff = np.angle(cross_spectrum)
    mean_phase_diff = np.mean(np.abs(phase_diff), axis=1)
    
    return {
        'mean_coherence_by_scale': mean_coherence.tolist(),
        'overall_coherence': float(np.mean(np.real(coherence))),
        'max_coherence_scale': int(scales[np.argmax(mean_coherence)]),
        'phase_diff_by_scale': mean_phase_diff.tolist(),
        'coherence_variability': float(np.std(mean_coherence)),
    }

# ════════════════════════════════════════════════════════════════════════════════
# COMPLETE CHOREOGRAPHY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def analyze_choreography(h1: np.ndarray, l1: np.ndarray) -> Dict:
    """Complete choreography analysis for a single window."""
    results = {}
    
    # 1. Symbolic Dynamics
    results['symbolic_h1'] = symbolic_dynamics_analysis(h1)
    results['symbolic_l1'] = symbolic_dynamics_analysis(l1)
    
    # 2. Recurrence Analysis
    results['recurrence_h1'] = recurrence_analysis(h1)
    results['recurrence_l1'] = recurrence_analysis(l1)
    
    # 3. Ordinal Patterns
    results['ordinal_h1'] = ordinal_patterns(h1)
    results['ordinal_l1'] = ordinal_patterns(l1)
    
    # 4. Information Flow
    results['information_flow'] = bidirectional_information_flow(h1, l1)
    
    # 5. Phase Space
    results['phase_space_h1'] = phase_space_analysis(h1)
    results['phase_space_l1'] = phase_space_analysis(l1)
    
    # 6. Visibility Graph
    results['visibility_h1'] = visibility_graph_metrics(h1)
    results['visibility_l1'] = visibility_graph_metrics(l1)
    
    # 7. Wavelet Analysis
    results['wavelet_h1'] = wavelet_analysis(h1)
    results['wavelet_l1'] = wavelet_analysis(l1)
    results['wavelet_coherence'] = wavelet_coherence(h1, l1)
    
    return results

def compare_choreographies(pre: Dict, post: Dict) -> Dict:
    """Compare choreography metrics between PRE and POST."""
    changes = {}
    
    # Key metrics to compare
    metrics = [
        ('symbolic_h1', 'word_entropy'),
        ('symbolic_h1', 'forbidden_ratio'),
        ('recurrence_h1', 'determinism'),
        ('recurrence_h1', 'laminarity'),
        ('ordinal_h1', 'permutation_entropy'),
        ('ordinal_h1', 'statistical_complexity'),
        ('information_flow', 'net_flow'),
        ('information_flow', 'total_information_flow'),
        ('phase_space_h1', 'correlation_dimension'),
        ('phase_space_h1', 'central_density'),
        ('visibility_h1', 'mean_degree'),
        ('visibility_h1', 'avg_clustering'),
        ('wavelet_h1', 'scale_entropy'),
        ('wavelet_coherence', 'overall_coherence'),
    ]
    
    for category, metric in metrics:
        try:
            pre_val = pre[category][metric]
            post_val = post[category][metric]
            
            if isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)):
                delta = post_val - pre_val
                rel_change = delta / (abs(pre_val) + 1e-10)
                
                changes[f"{category}.{metric}"] = {
                    'pre': pre_val,
                    'post': post_val,
                    'delta': delta,
                    'relative_change': rel_change,
                }
        except (KeyError, TypeError):
            continue
    
    # Sort by absolute relative change
    sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]['relative_change']), reverse=True)
    
    return {
        'all_changes': dict(sorted_changes),
        'top_changes': dict(sorted_changes[:10]),
        'n_metrics_compared': len(changes),
    }

# ════════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def analyze_event(event: EventConfig) -> Dict:
    """Complete choreography analysis for one event."""
    print(f"\n{'═'*70}")
    print(f"  🌍 {event.name} (M{event.magnitude}, {event.location})")
    print(f"{'═'*70}")
    
    h1, l1, eq_offset = load_ligo_data(event)
    
    if h1 is None:
        return {'error': 'Could not load data'}
    
    print(f"  ✅ Loaded {len(h1)} samples, offset={eq_offset}")
    
    results = {
        'event': event.name,
        'magnitude': event.magnitude,
        'location': event.location,
    }
    
    # Analyze each window
    windows_data = {}
    for window_name, (t_start, t_end) in WINDOWS.items():
        print(f"  🔬 Analyzing {window_name}...", end=" ", flush=True)
        
        h1_win = extract_window(h1, eq_offset, t_start, t_end)
        l1_win = extract_window(l1, eq_offset, t_start, t_end)
        
        if h1_win is None or l1_win is None:
            print("⚠️ Window too small")
            continue
        
        choreography = analyze_choreography(h1_win, l1_win)
        windows_data[window_name] = choreography
        print("✓")
    
    results['windows'] = windows_data
    
    # Compare choreographies
    print(f"  📊 Comparing choreographies...")
    
    if 'PRE' in windows_data and 'POST' in windows_data:
        results['pre_vs_post'] = compare_choreographies(windows_data['PRE'], windows_data['POST'])
        
        # Print top changes
        print(f"\n  🔝 Top choreography changes (PRE → POST):")
        for metric, data in list(results['pre_vs_post']['top_changes'].items())[:5]:
            print(f"      {metric}: {data['pre']:.4f} → {data['post']:.4f} ({data['relative_change']*100:+.1f}%)")
    
    if 'PRE' in windows_data and 'PRECURSOR' in windows_data:
        results['pre_vs_precursor'] = compare_choreographies(windows_data['PRE'], windows_data['PRECURSOR'])
        
        print(f"\n  🔝 Top choreography changes (PRE → PRECURSOR):")
        for metric, data in list(results['pre_vs_precursor']['top_changes'].items())[:5]:
            print(f"      {metric}: {data['pre']:.4f} → {data['post']:.4f} ({data['relative_change']*100:+.1f}%)")
    
    return results

def main():
    print("╔" + "═"*70 + "╗")
    print("║" + "  VACUUM TOMOGRAPHY v1.0".center(70) + "║")
    print("║" + "  'Scharfstellen auf die Choreographie'".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    start_time = datetime.now()
    
    all_results = []
    
    print(f"\n📂 Data: {DATA_DIR}")
    print(f"📊 Events: {len(EVENTS)}")
    print(f"🔬 Methods: Symbolic, Recurrence, Ordinal, Transfer Entropy, Phase Space, Visibility, Wavelet")
    
    for event in EVENTS:
        try:
            result = analyze_event(event)
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()
            all_results.append({'event': event.name, 'error': str(e)})
    
    # ════════════════════════════════════════════════════════════════════════════
    # CROSS-EVENT ANALYSIS
    # ════════════════════════════════════════════════════════════════════════════
    
    print("\n" + "═"*70)
    print("  📊 CROSS-EVENT CHOREOGRAPHY ANALYSIS")
    print("═"*70)
    
    # Aggregate changes across all events
    pre_post_changes = {}
    pre_precursor_changes = {}
    
    for result in all_results:
        if 'pre_vs_post' in result:
            for metric, data in result['pre_vs_post']['all_changes'].items():
                if metric not in pre_post_changes:
                    pre_post_changes[metric] = []
                pre_post_changes[metric].append(data['relative_change'])
        
        if 'pre_vs_precursor' in result:
            for metric, data in result['pre_vs_precursor']['all_changes'].items():
                if metric not in pre_precursor_changes:
                    pre_precursor_changes[metric] = []
                pre_precursor_changes[metric].append(data['relative_change'])
    
    # Statistical tests
    print("\n  🧪 Statistical Significance (t-test, H0: mean change = 0)")
    print("\n  PRE → PRECURSOR (IT FROM BIT):")
    print(f"  {'Metric':<45} {'Mean Δ':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*5}")
    
    precursor_results = []
    for metric, changes in sorted(pre_precursor_changes.items()):
        if len(changes) >= 3:
            t_stat, p_val = stats.ttest_1samp(changes, 0)
            mean_change = np.mean(changes)
            sig = "✓" if p_val < 0.05 else ""
            precursor_results.append((metric, mean_change, p_val, sig))
    
    precursor_results.sort(key=lambda x: x[2])
    for metric, mean_change, p_val, sig in precursor_results[:15]:
        print(f"  {metric:<45} {mean_change*100:>+9.1f}% {p_val:>10.4f} {sig:>5}")
    
    print("\n  PRE → POST (BIT FROM IT):")
    print(f"  {'Metric':<45} {'Mean Δ':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*5}")
    
    post_results = []
    for metric, changes in sorted(pre_post_changes.items()):
        if len(changes) >= 3:
            t_stat, p_val = stats.ttest_1samp(changes, 0)
            mean_change = np.mean(changes)
            sig = "✓" if p_val < 0.05 else ""
            post_results.append((metric, mean_change, p_val, sig))
    
    post_results.sort(key=lambda x: x[2])
    for metric, mean_change, p_val, sig in post_results[:15]:
        print(f"  {metric:<45} {mean_change*100:>+9.1f}% {p_val:>10.4f} {sig:>5}")
    
    # Summary
    n_precursor_sig = sum(1 for r in precursor_results if r[3] == "✓")
    n_post_sig = sum(1 for r in post_results if r[3] == "✓")
    
    print("\n" + "═"*70)
    print("  🎯 CHOREOGRAPHY SUMMARY")
    print("═"*70)
    print(f"""
  IT FROM BIT (PRE → PRECURSOR):
    Significant choreography changes: {n_precursor_sig}/{len(precursor_results)}
    
  BIT FROM IT (PRE → POST):
    Significant choreography changes: {n_post_sig}/{len(post_results)}
    
  INTERPRETATION:
    {'✅ CHOREOGRAPHY REVEALS BIT FROM IT!' if n_post_sig >= 3 else '❌ BIT FROM IT still not visible in choreography'}
    {'✅ IT FROM BIT confirmed in choreography!' if n_precursor_sig >= 3 else ''}
""")
    
    # Save results
    output_file = OUTPUT_DIR / f"vacuum_tomography_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        if isinstance(obj, tuple):
            return [convert_for_json(i) for i in obj]
        return obj
    
    final_results = {
        'metadata': {
            'version': '1.0',
            'n_events': len(EVENTS),
            'methods': ['symbolic_dynamics', 'recurrence', 'ordinal_patterns', 
                       'transfer_entropy', 'phase_space', 'visibility_graph', 'wavelet'],
            'runtime_seconds': (datetime.now() - start_time).total_seconds(),
        },
        'cross_event_statistics': {
            'pre_vs_precursor': {m: {'mean': np.mean(c), 'p_val': stats.ttest_1samp(c, 0)[1]} 
                                 for m, c in pre_precursor_changes.items() if len(c) >= 3},
            'pre_vs_post': {m: {'mean': np.mean(c), 'p_val': stats.ttest_1samp(c, 0)[1]} 
                           for m, c in pre_post_changes.items() if len(c) >= 3},
        },
        'events': all_results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(final_results), f, indent=2)
    
    elapsed = datetime.now() - start_time
    
    print(f"\n  ⏱️  Runtime: {elapsed}")
    print(f"  📁 Results: {output_file}")
    print("═"*70)

if __name__ == "__main__":

    main()
