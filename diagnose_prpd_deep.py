"""
Diagnóstico exhaustivo del PRPD para entender los límites físicos del dataset.
Evalúa:
1. Resolución de frecuencia con búsqueda ultra-fina (500k pasos)
2. Análisis por segmentos individuales del osciloscopio
3. Distribución angular de los clusters
4. Genera PRPD por segmento para verificar si el smearing es inter-segmento
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses

def main():
    folder = "Prueba 1 - Internas"
    channel = "CH3"
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    file_path = base_dir / folder / f"{channel}.csv"
    out_dir = base_dir / "DeltaPD_improved" / "outputs" / "debug_prpd"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    print(f"Total pulses: {len(toa_s)}")
    print(f"Time span: {toa_s[0]:.6f} to {toa_s[-1]:.6f} s ({toa_s[-1]-toa_s[0]:.2f} s total)")
    
    # ======= 1. Detect segments (gaps > 0.1s) =======
    dt = np.diff(toa_s)
    gap_threshold = 0.1  # 100ms
    gap_indices = np.where(dt > gap_threshold)[0]
    
    seg_starts = [0] + list(gap_indices + 1)
    seg_ends = list(gap_indices + 1) + [len(toa_s)]
    n_segments = len(seg_starts)
    
    print(f"\n=== SEGMENTS DETECTED (gap > {gap_threshold}s) ===")
    print(f"Number of segments: {n_segments}")
    for i, (s, e) in enumerate(zip(seg_starts, seg_ends)):
        n = e - s
        duration = toa_s[e-1] - toa_s[s] if n > 1 else 0
        print(f"  Seg {i}: pulses {s}-{e-1} ({n} pulses, {duration:.4f}s, t=[{toa_s[s]:.4f}, {toa_s[e-1]:.4f}])")
    
    # ======= 2. Ultra-fine frequency search =======
    print("\n=== ULTRA-FINE FREQUENCY SEARCH ===")
    f_test = np.linspace(49.5, 50.5, 500000)
    
    toa_sub = toa_s if len(toa_s) <= 10000 else np.random.choice(toa_s, 10000, replace=False)
    phase_matrix = 4 * np.pi * np.outer(f_test, toa_sub)
    z = np.exp(1j * phase_matrix)
    R = np.abs(np.mean(z, axis=1))
    
    # Find top 5 peaks
    from scipy.signal import find_peaks
    peaks_idx, props = find_peaks(R, height=0.3, distance=100)
    top_peaks = sorted(peaks_idx, key=lambda i: R[i], reverse=True)[:10]
    
    print("Top 10 R peaks:")
    for idx in top_peaks:
        print(f"  f={f_test[idx]:.6f} Hz, R={R[idx]:.4f}")
    
    best_f = f_test[np.argmax(R)]
    print(f"\nBest global: {best_f:.6f} Hz, R={np.max(R):.4f}")
    
    # Plot R(f) landscape
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(f_test, R, linewidth=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Kuramoto R")
    ax.set_title("Kuramoto Order Parameter Landscape (500k points)")
    ax.axvline(best_f, color='red', linestyle='--', label=f'Best: {best_f:.4f} Hz')
    ax.axvline(50.0, color='green', linestyle=':', label='50.0 Hz')
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(out_dir / "kuramoto_landscape.png"), dpi=150)
    plt.close()
    
    # ======= 3. Per-segment frequency and PRPD =======
    print("\n=== PER-SEGMENT ANALYSIS ===")
    fig, axes = plt.subplots(min(n_segments, 6), 1, figsize=(10, 4*min(n_segments, 6)))
    if n_segments == 1:
        axes = [axes]
    
    for i, (s, e) in enumerate(zip(seg_starts[:6], seg_ends[:6])):
        toa_seg = toa_s[s:e]
        peaks_seg = peaks[s:e]
        
        if len(toa_seg) < 10:
            print(f"  Seg {i}: too few pulses, skipping")
            continue
            
        # Local frequency search
        f_local = np.linspace(49.5, 50.5, 100000)
        z_local = np.exp(1j * 4 * np.pi * np.outer(f_local, toa_seg))
        R_local = np.abs(np.mean(z_local, axis=1))
        best_f_local = f_local[np.argmax(R_local)]
        
        # Phase with local best
        T = 1.0 / best_f_local
        phase_local = np.mod(toa_seg, T) / T * 360.0
        
        # Phase with global best
        T_global = 1.0 / best_f
        phase_global = np.mod(toa_seg, T_global) / T_global * 360.0
        
        # Circular variance
        theta_local = np.deg2rad(phase_local) * 2
        R_circ_local = np.abs(np.mean(np.exp(1j * theta_local)))
        theta_global = np.deg2rad(phase_global) * 2
        R_circ_global = np.abs(np.mean(np.exp(1j * theta_global)))
        
        print(f"  Seg {i}: local_f={best_f_local:.4f} Hz (R_circ={R_circ_local:.3f}), global_f R_circ={R_circ_global:.3f}")
        
        ax = axes[i]
        y_signed = np.where((phase_global >= 0) & (phase_global <= 180), peaks_seg, -peaks_seg)
        ax.scatter(phase_global, y_signed, s=5, alpha=0.7)
        ax.set_xlim(0, 360)
        ax.set_title(f"Seg {i}: {len(toa_seg)} pulses, f_local={best_f_local:.4f} Hz")
        ax.set_xlabel("Phase (deg)")
        ax.set_ylabel("Peak (V)")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(str(out_dir / "per_segment_prpd.png"), dpi=150)
    plt.close()
    
    # ======= 4. Angular histogram of phase distribution =======
    T_best = 1.0 / best_f
    phases_all = np.mod(toa_s, T_best) / T_best * 360.0
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(phases_all, bins=72, range=(0, 360), edgecolor='black', alpha=0.7)
    ax.set_xlabel("Phase (degrees)")
    ax.set_ylabel("Count")
    ax.set_title(f"Phase Distribution at f={best_f:.4f} Hz (N={len(phases_all)})")
    ax.set_xticks(np.arange(0, 361, 45))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_dir / "phase_histogram.png"), dpi=150)
    plt.close()
    
    # ======= 5. Phase width statistics =======
    # Find the two clusters using circular mean
    theta2 = np.deg2rad(phases_all) * 2
    mean_angle = np.arctan2(np.mean(np.sin(theta2)), np.mean(np.cos(theta2))) / 2.0
    center1 = np.rad2deg(mean_angle) % 360
    center2 = (center1 + 180) % 360
    
    # Assign pulses to nearest cluster
    d1 = np.minimum(np.abs(phases_all - center1), 360 - np.abs(phases_all - center1))
    d2 = np.minimum(np.abs(phases_all - center2), 360 - np.abs(phases_all - center2))
    cluster1 = phases_all[d1 < d2]
    cluster2 = phases_all[d1 >= d2]
    
    # Circular std of each cluster
    def circ_std(angles_deg):
        theta = np.deg2rad(angles_deg)
        R = np.abs(np.mean(np.exp(1j * theta)))
        return np.rad2deg(np.sqrt(-2 * np.log(R))) if R > 0 else 180
    
    std1 = circ_std(cluster1)
    std2 = circ_std(cluster2)
    
    print(f"\n=== CLUSTER ANALYSIS ===")
    print(f"Cluster 1: center={center1:.1f}°, N={len(cluster1)}, circ_std={std1:.1f}°")
    print(f"Cluster 2: center={center2:.1f}°, N={len(cluster2)}, circ_std={std2:.1f}°")
    print(f"Total phase spread (cluster1): {np.ptp(cluster1):.1f}° (min={np.min(cluster1):.1f}°, max={np.max(cluster1):.1f}°)")
    print(f"Total phase spread (cluster2): {np.ptp(cluster2):.1f}° (min={np.min(cluster2):.1f}°, max={np.max(cluster2):.1f}°)")
    
    print("\nAll debug plots saved to:", out_dir)

if __name__ == '__main__':
    main()
