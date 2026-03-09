import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import gaussian_kde

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.blind_prpd import reconstruct_blind_prpd

base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
folder = "Prueba 3 - Ensayo de Fuentes Múltiples Simultáneas"
channels = ["CH2", "CH3", "CH4"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, ch in enumerate(channels):
    file_path = base_dir / folder / f"{ch}.csv"
    print(f"Processing {ch}...")
    
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    phases, peaks_out = reconstruct_blind_prpd(toa_s[1:], peaks[1:], freq_hz=50.0, auto_calibrate=True)
    
    # Signed amplitude
    y = np.where((phases >= 0) & (phases <= 180), peaks_out, -peaks_out)
    
    # Outlier filtering for viz
    theta2 = np.deg2rad(phases) * 2
    mean_angle = np.arctan2(np.mean(np.sin(theta2)), np.mean(np.cos(theta2))) / 2.0
    c1 = np.rad2deg(mean_angle) % 360
    c2 = (c1 + 180) % 360
    d1 = np.minimum(np.abs(phases - c1), 360 - np.abs(phases - c1))
    d2 = np.minimum(np.abs(phases - c2), 360 - np.abs(phases - c2))
    d_min = np.minimum(d1, d2)
    med_d = np.median(d_min)
    mad = np.median(np.abs(d_min - med_d))
    thresh = med_d + 2.5 * max(mad * 1.4826, 5.0)
    inlier = d_min <= thresh
    
    ax = axes[i]
    
    # Gray outliers
    if np.sum(~inlier) > 0:
        ax.scatter(phases[~inlier], y[~inlier], c="lightgray", s=6, alpha=0.4, edgecolors="none", zorder=1)
    
    # KDE coloring
    xp, yp = phases[inlier], y[inlier]
    try:
        xy_s = np.vstack([xp, yp])
        kde = gaussian_kde(xy_s, bw_method=0.15)
        z = kde(xy_s)
        idx_sort = z.argsort()
        ax.scatter(xp[idx_sort], yp[idx_sort], c=z[idx_sort], cmap="turbo", s=12, alpha=0.9, edgecolors="none", zorder=2)
    except:
        ax.scatter(xp, yp, c="blue", s=12, alpha=0.7, edgecolors="none", zorder=2)
    
    # Sine ref
    t_sin = np.linspace(0, 360, 360)
    max_amp = np.max(np.abs(y)) * 1.05
    ax.plot(t_sin, max_amp * np.sin(np.radians(t_sin)), color="red", alpha=0.5, linewidth=1.5)
    
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_ylim(-max_amp * 1.15, max_amp * 1.15)
    ax.set_xlabel("Fase (grados)")
    ax.set_title(f"{ch} — {len(toa_s)} pulsos")
    ax.grid(True, linestyle=":", alpha=0.4)

axes[0].set_ylabel("Carga Aparente (V)")
fig.suptitle("PRPD Comparativo — Prueba 3 (Fuentes Múltiples) — CH2 vs CH3 vs CH4", fontsize=14, fontweight="bold")
plt.tight_layout()
out = str(base_dir / "DeltaPD_improved" / "outputs" / "prpd_p3_3ch_comparison.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {out}")
