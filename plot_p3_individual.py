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
out_dir = base_dir / "DeltaPD_improved" / "outputs"

for ch in channels:
    file_path = base_dir / folder / f"{ch}.csv"
    print(f"Processing {ch}...")
    
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    phases, peaks_out = reconstruct_blind_prpd(toa_s[1:], peaks[1:], freq_hz=50.0, auto_calibrate=True)
    y = np.where((phases >= 0) & (phases <= 180), peaks_out, -peaks_out)
    
    # Outlier filter
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
    inlier = d_min <= thresh
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    if np.sum(~inlier) > 0:
        ax.scatter(phases[~inlier], y[~inlier], c="lightgray", s=6, alpha=0.4, edgecolors="none", zorder=1)
    
    xp, yp = phases[inlier], y[inlier]
    try:
        sub = min(len(xp), 5000)
        idx_sub = np.random.choice(len(xp), sub, replace=False) if len(xp) > 5000 else np.arange(len(xp))
        kde = gaussian_kde(np.vstack([xp[idx_sub], yp[idx_sub]]), bw_method=0.15)
        z = kde(np.vstack([xp, yp]))
        s = z.argsort()
        ax.scatter(xp[s], yp[s], c=z[s], cmap="turbo", s=10, alpha=0.9, edgecolors="none", zorder=2)
    except:
        ax.scatter(xp, yp, c="blue", s=10, alpha=0.7, edgecolors="none", zorder=2)
    
    t_sin = np.linspace(0, 360, 360)
    ma2 = np.max(np.abs(y)) * 1.05
    ax.plot(t_sin, ma2 * np.sin(np.radians(t_sin)), color="red", alpha=0.6, linewidth=1.5)
    
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_ylim(-ma2 * 1.15, ma2 * 1.15)
    ax.set_xlabel("Fase (grados)")
    ax.set_ylabel("Carga Aparente (V)")
    ax.set_title(f"PRPD — Prueba 3 (Fuentes Múltiples) — {ch} — {len(toa_s)} pulsos")
    ax.grid(True, linestyle=":", alpha=0.4)
    
    out_png = str(out_dir / f"prpd_p3_{ch}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_png}")
