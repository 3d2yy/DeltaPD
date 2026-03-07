import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import gaussian_kde

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.blind_prpd import reconstruct_blind_prpd

def benchmark_prpd_renders():
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    folder = "Prueba 2 - Superficiales"
    channel = "CH3"
    file_path = base_dir / folder / f"{channel}.csv"
    out_dir = base_dir / "DeltaPD_improved" / "outputs" / "debug_prpd"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    phases, peaks_out = reconstruct_blind_prpd(toa_s[1:], peaks[1:], freq_hz=50.0, auto_calibrate=True)
    y = np.where((phases >= 0) & (phases <= 180), peaks_out, -peaks_out)
    
    # Filter Outliers
    theta2 = np.deg2rad(phases) * 2
    ma = np.arctan2(np.mean(np.sin(theta2)), np.mean(np.cos(theta2))) / 2.0
    c1 = np.rad2deg(ma) % 360
    c2 = (c1 + 180) % 360
    d1 = np.minimum(np.abs(phases - c1), 360 - np.abs(phases - c1))
    d2 = np.minimum(np.abs(phases - c2), 360 - np.abs(phases - c2))
    d_min = np.minimum(d1, d2)
    med_d = np.median(d_min)
    mad = np.median(np.abs(d_min - med_d))
    thresh = med_d + 2.5 * max(mad * 1.4826, 5.0)
    inliers = d_min <= thresh
    
    xp, yp = phases[inliers], y[inliers]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    t_sin = np.linspace(0, 360, 360)
    ma_sine = np.max(np.abs(y)) * 1.05
    
    # --- Option 1: Base KDE (Current Master) ---
    ax = axes[0]
    if np.sum(~inliers) > 0:
        ax.scatter(phases[~inliers], y[~inliers], c="lightgray", s=6, alpha=0.4, edgecolors="none")
    try:
        kde1 = gaussian_kde(np.vstack([xp, yp]), bw_method=0.15)
        z1 = kde1(np.vstack([xp, yp]))
        s1 = z1.argsort()
        ax.scatter(xp[s1], yp[s1], c=z1[s1], cmap="turbo", s=12, alpha=0.9, edgecolors="none")
    except:
        pass
    ax.set_title("Opcion 1: Base KDE (Densidad Pura)")
    
    # --- Option 2: Amplitude-Weighted KDE ---
    ax = axes[1]
    if np.sum(~inliers) > 0:
        ax.scatter(phases[~inliers], y[~inliers], c="lightgray", s=6, alpha=0.4, edgecolors="none")
    try:
        weights = np.abs(yp)
        kde2 = gaussian_kde(np.vstack([xp, yp]), bw_method=0.15, weights=weights)
        z2 = kde2(np.vstack([xp, yp]))
        s2 = z2.argsort()
        sizes = 8 + 25 * (np.abs(yp[s2]) / np.max(np.abs(yp[s2]))) # Size mapping
        ax.scatter(xp[s2], yp[s2], c=z2[s2], cmap="turbo", s=sizes, alpha=0.9, edgecolors="none")
    except:
        pass
    ax.set_title("Opcion 2: KDE Ponderado por Amplitud")
    
    # --- Option 3: Hexbin (2D Histogram) ---
    ax = axes[2]
    if np.sum(~inliers) > 0:
        ax.scatter(phases[~inliers], y[~inliers], c="lightgray", s=6, alpha=0.4, edgecolors="none")
    hb = ax.hexbin(xp, yp, gridsize=60, cmap="turbo", mincnt=1, alpha=0.9)
    # Add small dots on top for low-density areas
    ax.scatter(xp, yp, c="black", s=2, alpha=0.3, edgecolors="none")
    ax.set_title("Opcion 3: Histogram 2D (Hexbin)")
    
    for ax in axes:
        ax.plot(t_sin, ma_sine * np.sin(np.radians(t_sin)), color="red", alpha=0.4, linewidth=1.5)
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 90))
        ax.set_xlabel("Fase (grados)")
        ax.grid(True, linestyle=":", alpha=0.4)
        
    axes[0].set_ylabel("Carga Aparente (V)")
    plt.tight_layout()
    out = str(out_dir / "prpd_rendering_benchmark.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nBenchmark saved to {out}")

if __name__ == '__main__':
    benchmark_prpd_renders()
