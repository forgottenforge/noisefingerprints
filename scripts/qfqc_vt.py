#!/usr/bin/env python3
"""
Version: 1.0 - "The Wheeler Test"
"""

import numpy as np
import json
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except:
    HAS_MPL = False

try:
    from scipy import signal, stats
    from scipy.fft import fft, ifft
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except:
    HAS_SCIPY = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, confusion_matrix
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False
    print("⚠️ sklearn not found - some features disabled")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EarthquakeEvent:
    name: str
    gps_time: int
    magnitude: float
    depth_km: float
    latitude: float
    longitude: float
    mechanism: str  # strike-slip, subduction, normal, thrust
    region: str
    h1_file: str
    l1_file: str

# Extended event data with geographic properties
EVENTS = [
    EarthquakeEvent(
        name="Peru_M8.0",
        gps_time=1243173680,
        magnitude=8.0,
        depth_km=110,  # Deep subduction
        latitude=-5.812,
        longitude=-75.270,
        mechanism="subduction",
        region="South_America",
        h1_file="H-H1_GWOSC_O3a_4KHZ_R1-1243168768-4096.hdf5",
        l1_file="L-L1_GWOSC_O3a_4KHZ_R1-1243168768-4096.hdf5"
    ),
    EarthquakeEvent(
        name="Ridgecrest_M6.4",
        gps_time=1246408415,
        magnitude=6.4,
        depth_km=10,  # Shallow
        latitude=35.705,
        longitude=-117.506,
        mechanism="strike-slip",
        region="North_America",
        h1_file="H-H1_GWOSC_O3a_4KHZ_R1-1246400512-4096.hdf5",
        l1_file="L-L1_GWOSC_O3a_4KHZ_R1-1246400512-4096.hdf5"
    ),
    EarthquakeEvent(
        name="Ridgecrest_M7.1",
        gps_time=1246418393,
        magnitude=7.1,
        depth_km=8,  # Shallow
        latitude=35.770,
        longitude=-117.599,
        mechanism="strike-slip",
        region="North_America",
        h1_file="H-H1_GWOSC_O3a_4KHZ_R1-1246412800-4096.hdf5",
        l1_file="L-L1_GWOSC_O3a_4KHZ_R1-1246412800-4096.hdf5"
    ),
    EarthquakeEvent(
        name="Albania_M6.4",
        gps_time=1258862070,
        magnitude=6.4,
        depth_km=22,  # Medium
        latitude=41.514,
        longitude=19.526,
        mechanism="thrust",
        region="Europe",
        h1_file="H-H1_GWOSC_O3b_4KHZ_R1-1258856448-4096.hdf5",
        l1_file="L-L1_GWOSC_O3b_4KHZ_R1-1258856448-4096.hdf5"
    ),
]

# Analysis parameters
SR = 64
N_WORKERS = mp.cpu_count()
N_PHASE_BINS = 36  # 10° resolution for phase histogram
N_FREQ_BANDS = 8   # Frequency bands for spectral phase
N_BOOTSTRAP = 100  # For signature stability

DATA_DIR = Path("data")
OUTPUT_DIR = Path("vacuum_telescope")
OUTPUT_DIR.mkdir(exist_ok=True)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔭 QFQC VACUUM TELESCOPE v1.0 🔭                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  "We don't look AT earthquakes - we read their ADDRESS in the vacuum"        ║
║                                                                              ║
║  The Question: Does each earthquake have a unique fingerprint?               ║
║  The Test: Can we classify events by their vacuum signature alone?           ║
║  The Prize: Proof that vacuum contains geographic information!               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# PHASE SIGNATURE EXTRACTION (The Core of the Vacuum Telescope)
# ============================================================================

def extract_phase_signature(h1: np.ndarray, l1: np.ndarray) -> Dict:
    """
    Extract the PHASE RELATIONSHIP signature between H1 and L1.
    
    This is the "address" in vacuum space - the unique fingerprint
    of how information is encoded in the detector correlation.
    """
    n = min(len(h1), len(l1))
    h1, l1 = h1[:n], l1[:n]
    
    signature = {}
    
    # 1. INSTANTANEOUS PHASE DIFFERENCE
    # The moment-by-moment phase relationship
    h1_analytic = signal.hilbert(h1) if HAS_SCIPY else h1 + 1j * np.imag(np.fft.fft(h1))
    l1_analytic = signal.hilbert(l1) if HAS_SCIPY else l1 + 1j * np.imag(np.fft.fft(l1))
    
    phase_h1 = np.angle(h1_analytic)
    phase_l1 = np.angle(l1_analytic)
    phase_diff = np.mod(phase_h1 - phase_l1 + np.pi, 2*np.pi) - np.pi
    
    # Phase difference histogram (the "fingerprint")
    phase_hist, _ = np.histogram(phase_diff, bins=N_PHASE_BINS, range=(-np.pi, np.pi), density=True)
    signature['phase_histogram'] = phase_hist
    
    # Phase statistics
    signature['phase_mean'] = float(np.mean(phase_diff))
    signature['phase_std'] = float(np.std(phase_diff))
    signature['phase_skew'] = float(stats.skew(phase_diff)) if HAS_SCIPY else 0
    signature['phase_kurtosis'] = float(stats.kurtosis(phase_diff)) if HAS_SCIPY else 0
    
    # 2. SPECTRAL PHASE COHERENCE
    # How phase relationship varies across frequencies
    if HAS_SCIPY:
        freqs, Cxy = signal.coherence(h1, l1, fs=SR, nperseg=min(512, n//4))
        _, Pxy = signal.csd(h1, l1, fs=SR, nperseg=min(512, n//4))
        
        spectral_phase = np.angle(Pxy)
        
        # Divide into frequency bands
        band_edges = np.logspace(np.log10(1), np.log10(SR/2), N_FREQ_BANDS + 1)
        band_coherence = []
        band_phase = []
        
        for i in range(N_FREQ_BANDS):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
            if np.any(mask):
                band_coherence.append(float(np.mean(Cxy[mask])))
                band_phase.append(float(np.mean(spectral_phase[mask])))
            else:
                band_coherence.append(0)
                band_phase.append(0)
        
        signature['spectral_coherence'] = band_coherence
        signature['spectral_phase'] = band_phase
    else:
        signature['spectral_coherence'] = [0] * N_FREQ_BANDS
        signature['spectral_phase'] = [0] * N_FREQ_BANDS
    
    # 3. MUTUAL INFORMATION PROFILE
    # How information is shared at different time lags
    max_lag = min(100, n // 10)
    mi_profile = []
    
    for lag in range(-max_lag, max_lag + 1, max_lag // 5):
        if lag < 0:
            h1_shifted = h1[-lag:]
            l1_shifted = l1[:lag]
        elif lag > 0:
            h1_shifted = h1[:-lag]
            l1_shifted = l1[lag:]
        else:
            h1_shifted, l1_shifted = h1, l1
        
        if len(h1_shifted) > 100:
            # Quick MI estimate
            hist2d, _, _ = np.histogram2d(h1_shifted, l1_shifted, bins=20)
            pxy = hist2d / hist2d.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            
            # MI = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
            mi = 0
            for i in range(len(px)):
                for j in range(len(py)):
                    if pxy[i,j] > 0 and px[i] > 0 and py[j] > 0:
                        mi += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[j]))
            mi_profile.append(float(mi))
        else:
            mi_profile.append(0)
    
    signature['mi_profile'] = mi_profile
    signature['mi_peak_lag'] = int(np.argmax(mi_profile) - len(mi_profile)//2)
    signature['mi_asymmetry'] = float(np.mean(mi_profile[len(mi_profile)//2:]) - 
                                       np.mean(mi_profile[:len(mi_profile)//2]))
    
    # 4. ENVELOPE CORRELATION STRUCTURE
    env_h1 = np.abs(h1_analytic)
    env_l1 = np.abs(l1_analytic)
    
    signature['envelope_corr'] = float(np.corrcoef(env_h1, env_l1)[0,1])
    signature['envelope_ratio'] = float(np.mean(env_h1) / (np.mean(env_l1) + 1e-15))
    
    # 5. PHASE LOCKING VALUE (PLV)
    # Measure of phase synchronization
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    signature['plv'] = float(plv)
    
    # 6. WEIGHTED PHASE LAG INDEX (wPLI)
    # Robust measure of phase lead/lag
    imag_part = np.imag(h1_analytic * np.conj(l1_analytic))
    wpli = np.abs(np.mean(np.sign(imag_part) * np.abs(imag_part))) / (np.mean(np.abs(imag_part)) + 1e-15)
    signature['wpli'] = float(wpli)
    
    return signature

def signature_to_vector(signature: Dict) -> np.ndarray:
    """Convert signature dict to feature vector for ML."""
    features = []
    
    # Phase histogram (36 features)
    features.extend(signature.get('phase_histogram', [0]*N_PHASE_BINS))
    
    # Phase statistics (4 features)
    features.append(signature.get('phase_mean', 0))
    features.append(signature.get('phase_std', 0))
    features.append(signature.get('phase_skew', 0))
    features.append(signature.get('phase_kurtosis', 0))
    
    # Spectral features (16 features)
    features.extend(signature.get('spectral_coherence', [0]*N_FREQ_BANDS))
    features.extend(signature.get('spectral_phase', [0]*N_FREQ_BANDS))
    
    # MI profile (11 features)
    features.extend(signature.get('mi_profile', [0]*11))
    features.append(signature.get('mi_peak_lag', 0))
    features.append(signature.get('mi_asymmetry', 0))
    
    # Envelope features (2 features)
    features.append(signature.get('envelope_corr', 0))
    features.append(signature.get('envelope_ratio', 0))
    
    # Synchronization features (2 features)
    features.append(signature.get('plv', 0))
    features.append(signature.get('wpli', 0))
    
    return np.array(features, dtype=np.float64)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_event_data(event: EarthquakeEvent) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Load LIGO data for an event."""
    h1_path = DATA_DIR / event.h1_file
    l1_path = DATA_DIR / event.l1_file
    
    if not h1_path.exists() or not l1_path.exists():
        return None, None, 0
    
    try:
        with h5py.File(h1_path, 'r') as f:
            h1 = f['strain']['Strain'][:]
        with h5py.File(l1_path, 'r') as f:
            l1 = f['strain']['Strain'][:]
        gps_start = int(event.h1_file.split('-')[-2])
        return h1, l1, gps_start
    except:
        return None, None, 0

def extract_window(strain: np.ndarray, gps_center: int, gps_start: int,
                   window_sec: Tuple[int, int]) -> Optional[np.ndarray]:
    """Extract time window from strain data."""
    start = int((gps_center + window_sec[0] - gps_start) * SR)
    end = int((gps_center + window_sec[1] - gps_start) * SR)
    if start < 0 or end > len(strain):
        return None
    data = strain[start:end]
    return data if len(data) >= 100 and np.all(np.isfinite(data)) else None

# ============================================================================
# VACUUM TELESCOPE ANALYSIS
# ============================================================================

def analyze_event_signature(event: EarthquakeEvent) -> Optional[Dict]:
    """Extract complete vacuum signature for an event."""
    
    print(f"\n  🔭 Scanning {event.name} (M{event.magnitude}, {event.region})...")
    
    h1_full, l1_full, gps_start = load_event_data(event)
    if h1_full is None:
        print(f"     ❌ Data not found")
        return None
    
    # Extract POST window (where vacuum "remembers" the event)
    h1 = extract_window(h1_full, event.gps_time, gps_start, (60, 300))
    l1 = extract_window(l1_full, event.gps_time, gps_start, (60, 300))
    
    if h1 is None or l1 is None:
        print(f"     ❌ Insufficient data")
        return None
    
    # Extract signature
    signature = extract_phase_signature(h1, l1)
    
    # Add geographic metadata
    signature['event_name'] = event.name
    signature['magnitude'] = event.magnitude
    signature['depth_km'] = event.depth_km
    signature['latitude'] = event.latitude
    signature['longitude'] = event.longitude
    signature['mechanism'] = event.mechanism
    signature['region'] = event.region
    
    # Bootstrap for stability estimate
    print(f"     📊 Bootstrap stability test...")
    bootstrap_vectors = []
    n = len(h1)
    
    for i in range(N_BOOTSTRAP):
        idx = np.random.randint(0, n, n)
        h1_boot = h1[idx]
        l1_boot = l1[idx]
        sig_boot = extract_phase_signature(h1_boot, l1_boot)
        bootstrap_vectors.append(signature_to_vector(sig_boot))
    
    bootstrap_vectors = np.array(bootstrap_vectors)
    signature['stability'] = float(1.0 / (np.mean(np.std(bootstrap_vectors, axis=0)) + 1e-15))
    
    print(f"     ✅ Signature extracted (stability: {signature['stability']:.2f})")
    
    return signature

def compute_signature_distances(signatures: List[Dict]) -> np.ndarray:
    """Compute pairwise distances between event signatures."""
    
    vectors = []
    for sig in signatures:
        vectors.append(signature_to_vector(sig))
    
    vectors = np.array(vectors)
    
    # Standardize
    if HAS_SKLEARN:
        scaler = StandardScaler()
        vectors = scaler.fit_transform(vectors)
    
    # Compute distance matrix
    distances = pdist(vectors, metric='euclidean')
    distance_matrix = squareform(distances)
    
    return distance_matrix, vectors

# ============================================================================
# GEOGRAPHIC DECODING (Can we read location from vacuum?)
# ============================================================================

def test_geographic_decoding(signatures: List[Dict], vectors: np.ndarray) -> Dict:
    """Test if we can decode geographic properties from vacuum signatures."""
    
    results = {}
    
    # Extract geographic labels
    magnitudes = np.array([s['magnitude'] for s in signatures])
    depths = np.array([s['depth_km'] for s in signatures])
    latitudes = np.array([s['latitude'] for s in signatures])
    longitudes = np.array([s['longitude'] for s in signatures])
    mechanisms = [s['mechanism'] for s in signatures]
    regions = [s['region'] for s in signatures]
    
    # 1. MAGNITUDE CORRELATION
    # Can vacuum signature predict earthquake magnitude?
    if HAS_SKLEARN and len(signatures) >= 3:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import LeaveOneOut
        
        loo = LeaveOneOut()
        mag_predictions = []
        mag_actual = []
        
        for train_idx, test_idx in loo.split(vectors):
            model = LinearRegression()
            model.fit(vectors[train_idx], magnitudes[train_idx])
            pred = model.predict(vectors[test_idx])
            mag_predictions.append(pred[0])
            mag_actual.append(magnitudes[test_idx][0])
        
        mag_corr = np.corrcoef(mag_actual, mag_predictions)[0,1]
        results['magnitude_correlation'] = float(mag_corr)
        results['magnitude_predictions'] = list(zip(mag_actual, mag_predictions))
    
    # 2. DEPTH CORRELATION
    if HAS_SKLEARN and len(signatures) >= 3:
        loo = LeaveOneOut()
        depth_predictions = []
        depth_actual = []
        
        for train_idx, test_idx in loo.split(vectors):
            model = LinearRegression()
            model.fit(vectors[train_idx], depths[train_idx])
            pred = model.predict(vectors[test_idx])
            depth_predictions.append(pred[0])
            depth_actual.append(depths[test_idx][0])
        
        depth_corr = np.corrcoef(depth_actual, depth_predictions)[0,1]
        results['depth_correlation'] = float(depth_corr)
    
    # 3. LOCATION CORRELATION
    if HAS_SKLEARN and len(signatures) >= 3:
        # Combine lat/lon into distance from reference point
        ref_lat, ref_lon = 0, 0  # Equator/Prime Meridian
        distances_from_ref = np.sqrt((latitudes - ref_lat)**2 + (longitudes - ref_lon)**2)
        
        loo = LeaveOneOut()
        loc_predictions = []
        loc_actual = []
        
        for train_idx, test_idx in loo.split(vectors):
            model = LinearRegression()
            model.fit(vectors[train_idx], distances_from_ref[train_idx])
            pred = model.predict(vectors[test_idx])
            loc_predictions.append(pred[0])
            loc_actual.append(distances_from_ref[test_idx][0])
        
        loc_corr = np.corrcoef(loc_actual, loc_predictions)[0,1]
        results['location_correlation'] = float(loc_corr)
    
    # 4. MECHANISM CLASSIFICATION
    # Can we identify earthquake type from vacuum signature?
    unique_mechanisms = list(set(mechanisms))
    if len(unique_mechanisms) > 1 and HAS_SKLEARN:
        from sklearn.neighbors import KNeighborsClassifier
        
        mechanism_labels = [unique_mechanisms.index(m) for m in mechanisms]
        
        # Leave-one-out classification
        correct = 0
        for i in range(len(signatures)):
            train_idx = [j for j in range(len(signatures)) if j != i]
            test_idx = [i]
            
            knn = KNeighborsClassifier(n_neighbors=min(2, len(train_idx)))
            knn.fit(vectors[train_idx], [mechanism_labels[j] for j in train_idx])
            pred = knn.predict(vectors[test_idx])
            
            if pred[0] == mechanism_labels[i]:
                correct += 1
        
        results['mechanism_accuracy'] = float(correct / len(signatures))
    
    # 5. REGION CLASSIFICATION
    unique_regions = list(set(regions))
    if len(unique_regions) > 1 and HAS_SKLEARN:
        from sklearn.neighbors import KNeighborsClassifier
        
        region_labels = [unique_regions.index(r) for r in regions]
        
        correct = 0
        for i in range(len(signatures)):
            train_idx = [j for j in range(len(signatures)) if j != i]
            test_idx = [i]
            
            knn = KNeighborsClassifier(n_neighbors=min(2, len(train_idx)))
            knn.fit(vectors[train_idx], [region_labels[j] for j in train_idx])
            pred = knn.predict(vectors[test_idx])
            
            if pred[0] == region_labels[i]:
                correct += 1
        
        results['region_accuracy'] = float(correct / len(signatures))
        results['regions'] = unique_regions
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_vacuum_telescope_figures(signatures: List[Dict], vectors: np.ndarray, 
                                    distance_matrix: np.ndarray, decoding: Dict):
    """Create visualization of vacuum information space."""
    
    if not HAS_MPL:
        return
    
    print("\n  📊 Creating Vacuum Telescope visualizations...")
    
    # Figure 1: Vacuum Information Space (2D projection)
    fig1, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA projection
    if HAS_SKLEARN:
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(vectors)
        
        ax = axes[0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(signatures)))
        
        for i, sig in enumerate(signatures):
            ax.scatter(coords_pca[i, 0], coords_pca[i, 1], 
                      c=[colors[i]], s=sig['magnitude']**2 * 20, 
                      label=f"{sig['event_name']} (M{sig['magnitude']})",
                      edgecolors='black', linewidths=2)
            ax.annotate(sig['event_name'][:10], (coords_pca[i, 0], coords_pca[i, 1]),
                       fontsize=8, ha='center', va='bottom')
        
        ax.set_xlabel(f'Vacuum PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'Vacuum PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Vacuum Information Space (PCA)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Distance matrix heatmap
    ax = axes[1]
    im = ax.imshow(distance_matrix, cmap='viridis')
    ax.set_xticks(range(len(signatures)))
    ax.set_yticks(range(len(signatures)))
    ax.set_xticklabels([s['event_name'][:10] for s in signatures], rotation=45, ha='right')
    ax.set_yticklabels([s['event_name'][:10] for s in signatures])
    plt.colorbar(im, ax=ax, label='Vacuum Distance')
    ax.set_title('Earthquake Signature Distances')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_vacuum_space.png", dpi=150)
    plt.close()
    
    # Figure 2: Phase Fingerprints
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Phase histograms
    ax = axes[0, 0]
    for i, sig in enumerate(signatures):
        phase_hist = sig.get('phase_histogram', [])
        if len(phase_hist) > 0:
            x = np.linspace(-180, 180, len(phase_hist))
            ax.plot(x, phase_hist, label=sig['event_name'][:12], linewidth=2)
    ax.set_xlabel('Phase Difference (degrees)')
    ax.set_ylabel('Density')
    ax.set_title('Phase Fingerprints')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spectral coherence profiles
    ax = axes[0, 1]
    for i, sig in enumerate(signatures):
        coh = sig.get('spectral_coherence', [])
        if len(coh) > 0:
            ax.plot(range(len(coh)), coh, 'o-', label=sig['event_name'][:12], linewidth=2)
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Coherence')
    ax.set_title('Spectral Coherence Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MI profiles
    ax = axes[1, 0]
    for i, sig in enumerate(signatures):
        mi = sig.get('mi_profile', [])
        if len(mi) > 0:
            x = np.linspace(-1, 1, len(mi))
            ax.plot(x, mi, label=sig['event_name'][:12], linewidth=2)
    ax.set_xlabel('Time Lag (normalized)')
    ax.set_ylabel('Mutual Information')
    ax.set_title('Information Flow Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Geographic decoding results
    ax = axes[1, 1]
    metrics = ['magnitude_correlation', 'depth_correlation', 'location_correlation', 
               'mechanism_accuracy', 'region_accuracy']
    values = [decoding.get(m, 0) for m in metrics]
    labels = ['Magnitude\nCorr.', 'Depth\nCorr.', 'Location\nCorr.', 
              'Mechanism\nAccuracy', 'Region\nAccuracy']
    
    colors = ['green' if v > 0.5 else 'orange' if v > 0 else 'red' for v in values]
    bars = ax.bar(range(len(metrics)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Geographic Decoding from Vacuum Signature')
    ax.axhline(0.5, color='gray', linestyle='--', label='Chance level')
    ax.set_ylim(-1, 1)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_fingerprints.png", dpi=150)
    plt.close()
    
    # Figure 3: Dendrogram (hierarchical clustering)
    if HAS_SCIPY:
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        linkage_matrix = linkage(vectors, method='ward')
        dendrogram(linkage_matrix, labels=[s['event_name'][:12] for s in signatures],
                   ax=ax, leaf_rotation=45)
        ax.set_ylabel('Vacuum Distance')
        ax.set_title('Hierarchical Clustering of Earthquake Vacuum Signatures')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig3_dendrogram.png", dpi=150)
        plt.close()
    
    print(f"  ✅ Figures saved to {OUTPUT_DIR}/")

# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = datetime.now()
    
    print("\n" + "═" * 80)
    print("  PHASE 1: SIGNATURE EXTRACTION")
    print("═" * 80)
    
    signatures = []
    for event in EVENTS:
        sig = analyze_event_signature(event)
        if sig:
            signatures.append(sig)
    
    if len(signatures) < 2:
        print("\n  ❌ Need at least 2 events for analysis")
        return
    
    print(f"\n  ✅ Extracted signatures for {len(signatures)} events")
    
    # Compute distances
    print("\n" + "═" * 80)
    print("  PHASE 2: VACUUM SPACE ANALYSIS")
    print("═" * 80)
    
    distance_matrix, vectors = compute_signature_distances(signatures)
    
    print("\n  📐 Signature Distance Matrix:")
    print(f"  {'':20}", end="")
    for sig in signatures:
        print(f"{sig['event_name'][:10]:>12}", end="")
    print()
    
    for i, sig in enumerate(signatures):
        print(f"  {sig['event_name'][:20]:20}", end="")
        for j in range(len(signatures)):
            print(f"{distance_matrix[i,j]:>12.2f}", end="")
        print()
    
    # Geographic decoding test
    print("\n" + "═" * 80)
    print("  PHASE 3: GEOGRAPHIC DECODING TEST")
    print("═" * 80)
    print("\n  Question: Can we read geographic information from vacuum signatures?")
    
    decoding = test_geographic_decoding(signatures, vectors)
    
    print("\n  📊 DECODING RESULTS:")
    print(f"  {'─'*60}")
    
    if 'magnitude_correlation' in decoding:
        r = decoding['magnitude_correlation']
        status = "✅" if abs(r) > 0.5 else "⚠️"
        print(f"  {status} Magnitude prediction:  r = {r:+.3f}")
    
    if 'depth_correlation' in decoding:
        r = decoding['depth_correlation']
        status = "✅" if abs(r) > 0.5 else "⚠️"
        print(f"  {status} Depth prediction:      r = {r:+.3f}")
    
    if 'location_correlation' in decoding:
        r = decoding['location_correlation']
        status = "✅" if abs(r) > 0.5 else "⚠️"
        print(f"  {status} Location prediction:   r = {r:+.3f}")
    
    if 'mechanism_accuracy' in decoding:
        acc = decoding['mechanism_accuracy']
        status = "✅" if acc > 0.5 else "⚠️"
        print(f"  {status} Mechanism classification: {acc*100:.1f}% accuracy")
    
    if 'region_accuracy' in decoding:
        acc = decoding['region_accuracy']
        status = "✅" if acc > 0.5 else "⚠️"
        print(f"  {status} Region classification:    {acc*100:.1f}% accuracy")
    
    # Create visualizations
    create_vacuum_telescope_figures(signatures, vectors, distance_matrix, decoding)
    
    # Save results
    results = {
        'metadata': {
            'version': 'Vacuum Telescope v1.0',
            'timestamp': datetime.now().isoformat(),
            'n_events': len(signatures)
        },
        'signatures': signatures,
        'distance_matrix': distance_matrix.tolist(),
        'decoding_results': decoding
    }
    
    with open(OUTPUT_DIR / "vacuum_telescope_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Final verdict
    elapsed = datetime.now() - start_time
    
    print("\n" + "═" * 80)
    print("  🔭 VACUUM TELESCOPE - FINAL VERDICT")
    print("═" * 80)
    
    # Calculate overall score
    scores = []
    if 'magnitude_correlation' in decoding:
        scores.append(abs(decoding['magnitude_correlation']))
    if 'depth_correlation' in decoding:
        scores.append(abs(decoding['depth_correlation']))
    if 'location_correlation' in decoding:
        scores.append(abs(decoding['location_correlation']))
    if 'mechanism_accuracy' in decoding:
        scores.append(decoding['mechanism_accuracy'])
    if 'region_accuracy' in decoding:
        scores.append(decoding['region_accuracy'])
    
    avg_score = np.mean(scores) if scores else 0
    
    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  WHEELER'S "IT FROM BIT" HYPOTHESIS TEST                          │
  ├────────────────────────────────────────────────────────────────────┤
  │                                                                    │
  │  Events analyzed: {len(signatures)}                                            │
  │  Feature dimensions: {vectors.shape[1] if len(vectors.shape) > 1 else 'N/A'}                                          │
  │  Average decoding score: {avg_score:.3f}                                   │
  │                                                                    │""")
    
    if avg_score > 0.6:
        print("""  │  🔥🔥🔥 STRONG EVIDENCE: Vacuum contains geographic information!   │
  │                                                                    │
  │  The quantum vacuum appears to encode earthquake properties.       │
  │  Different events have DISTINGUISHABLE signatures.                 │
  │  This supports Wheeler's "It from Bit" hypothesis!                 │""")
    elif avg_score > 0.3:
        print("""  │  🔍 MODERATE EVIDENCE: Some geographic encoding detected          │
  │                                                                    │
  │  Vacuum signatures show partial correlation with geography.        │
  │  More events needed for confirmation.                              │""")
    else:
        print("""  │  ❓ INCONCLUSIVE: Limited geographic encoding                     │
  │                                                                    │
  │  Earthquake signatures are distinct but geographic decoding        │
  │  is limited. May need more events or refined methods.              │""")
    
    print(f"""  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
  
  ⏱️  Runtime: {elapsed}
  💾 Results: {OUTPUT_DIR}/vacuum_telescope_results.json
  📊 Figures: {OUTPUT_DIR}/fig1_vacuum_space.png
              {OUTPUT_DIR}/fig2_fingerprints.png
              {OUTPUT_DIR}/fig3_dendrogram.png
  
  "The vacuum is not empty - it is the universe's memory."
  "Every earthquake leaves an address. We are learning to read it." 🔭
""")

if __name__ == "__main__":

    main()
