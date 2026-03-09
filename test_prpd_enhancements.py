import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import gaussian_kde

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.blind_prpd import reconstruct_blind_prpd

def test_prpd_improvements():
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    folder = "Prueba 1 - Internas"
    channel = "CH3"
    file_path = base_dir / folder / f"{channel}.csv"
    out_dir = base_dir / "DeltaPD_improved" / "outputs" / "debug_prpd"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    
    # --- Experiment 1: Lower Threshold (3.5 sigma instead of 5.0) ---
    print("\nExperiment 1: Lower Threshold (3.5 sigma)")
    pulse_indices_low = detect_pulses(x, fs, threshold_sigma=3.5, min_separation_s=20e-9, method="threshold")
    toa_s_low = times_abs[pulse_indices_low]
    peaks_low = np.abs(x[pulse_indices_low])
    print(f"  Detected {len(toa_s_low)} pulses (vs ~660 at 5.0 sigma)")
    
    phases_low, peaks_out_low = reconstruct_blind_prpd(toa_s_low[1:], peaks_low[1:], freq_hz=50.0, auto_calibrate=True)
    y_low = np.where((phases_low >= 0) & (phases_low <= 180), peaks_out_low, -peaks_out_low)
    
    # --- Filter Outliers for Viz ---
    def get_inliers(phases):
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
        return d_min <= thresh

    inliers_low = get_inliers(phases_low)
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # 1. Plot Lower Threshold
    ax = axes[0]
    if np.sum(~inliers_low) > 0:
         ax.scatter(phases_low[~inliers_low], y_low[~inliers_low], c="lightgray", s=6, alpha=0.4, edgecolors="none")
    
    xp, yp = phases_low[inliers_low], y_low[inliers_low]
    try:
        sub = min(len(xp), 5000)
        idx_sub = np.random.choice(len(xp), sub, replace=False) if len(xp) > 5000 else np.arange(len(xp))
        kde = gaussian_kde(np.vstack([xp[idx_sub], yp[idx_sub]]), bw_method=0.15)
        z = kde(np.vstack([xp, yp]))
        s = z.argsort()
        ax.scatter(xp[s], yp[s], c=z[s], cmap="turbo", s=8, alpha=0.8, edgecolors="none")
    except Exception as e:
        print(f"KDE failed: {e}")
        ax.scatter(xp, yp, c="blue", s=8, alpha=0.7)
        
    ax.set_title(f"Lower Threshold (3.5 IQR) - {len(toa_s_low)} pulses")
    
    # --- Experiment 2: Amplitude-Weighted KDE (on 5.0 sigma data) ---
    print("\nExperiment 2: Amplitude-Weighted KDE (5.0 sigma)")
    pulse_indices_base = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s_base = times_abs[pulse_indices_base]
    peaks_base = np.abs(x[pulse_indices_base])
    
    phases_base, peaks_out_base = reconstruct_blind_prpd(toa_s_base[1:], peaks_base[1:], freq_hz=50.0, auto_calibrate=True)
    y_base = np.where((phases_base >= 0) & (phases_base <= 180), peaks_out_base, -peaks_out_base)
    inliers_base = get_inliers(phases_base)
    
    ax = axes[1]
    if np.sum(~inliers_base) > 0:
         ax.scatter(phases_base[~inliers_base], y_base[~inliers_base], c="lightgray", s=6, alpha=0.4, edgecolors="none")
         
    xp2, yp2 = phases_base[inliers_base], y_base[inliers_base]
    weights = np.abs(yp2)  # Weight KDE by amplitude
    
    try:
        kde2 = gaussian_kde(np.vstack([xp2, yp2]), bw_method=0.15, weights=weights)
        z2 = kde2(np.vstack([xp2, yp2]))
        s2 = z2.argsort()
        # Scale dot size by amplitude too for visual emphasis
        sizes = 5 + 30 * (np.abs(yp2[s2]) / np.max(np.abs(yp2[s2]))) 
        ax.scatter(xp2[s2], yp2[s2], c=z2[s2], cmap="turbo", s=sizes, alpha=0.9, edgecolors="none")
    except Exception as e:
         print(f"KDE failed: {e}")
         ax.scatter(xp2, yp2, c="blue", s=10)
         
    ax.set_title(f"Amplitude-Weighted KDE (5.0 IQR) - {len(toa_s_base)} pulses")
    
    for ax in axes:
        t_sin = np.linspace(0, 360, 360)
        ma2 = np.max(np.abs(y_low)) * 1.05
        ax.plot(t_sin, ma2 * np.sin(np.radians(t_sin)), color="red", alpha=0.4, linewidth=1.5)
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_xlabel("Fase (grados)")
        ax.grid(True, linestyle=":", alpha=0.4)
        
    axes[0].set_ylabel("Carga Aparente (V)")
    plt.tight_layout()
    out = str(out_dir / "prpd_improvements.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved test plot to {out}")

if __name__ == '__main__':
    test_prpd_improvements()
