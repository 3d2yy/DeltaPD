import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
import sys
from scipy.stats import gaussian_kde

sys.path.insert(0, 'e:/Carpeta definitiva de Tesis/programas/DeltaPD_improved/src')
from deltapd.descriptors import detect_pulses
from deltapd.blind_prpd import reconstruct_blind_prpd

def plot_mat_prpd():
    mat_file = Path("e:/Carpeta definitiva de Tesis/programas/DeltaPD-main/SignalTestEnvolpe01.mat")
    out_dir = Path("e:/Carpeta definitiva de Tesis/programas/DeltaPD_improved/outputs")
    
    print(f"Loading {mat_file}...")
    mat = sio.loadmat(str(mat_file))
    
    # Extract signal and sampling frequency
    # We assume 'signal' and 'fs' or similar are in the .mat file. Checking standard names:
    if 'y' in mat: x = mat['y'].ravel()
    elif 'val' in mat: x = mat['val'].ravel()
    elif 'Data' in mat: x = mat['Data'].ravel()
    elif 'Signal' in mat: x = mat['Signal'].ravel()
    else:
        # Just grab the largest numpy array
        arrays = [(k, v) for k, v in mat.items() if isinstance(v, np.ndarray)]
        arrays.sort(key=lambda item: item[1].size, reverse=True)
        x = arrays[0][1].ravel()
        
    fs = mat.get('fs', mat.get('Fs', 10e6)) # Fallback to 10MS/s if unknown
    if isinstance(fs, np.ndarray): fs = fs.item()
    
    print(f"Loaded signal of length {len(x)} with fs={fs} Hz")
    
    # 1. Detect Pulses (we might need a higher threshold for this specific file, we'll try 5.0 first)
    print("Detecting pulses...")
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=1e-5, method="threshold")
    times_abs = np.arange(len(x)) / fs
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    print(f"Detected {len(toa_s)} pulses.")
    
    if len(toa_s) < 10:
        print("Not enough pulses detected for PRPD. Try lowering threshold.")
        return
        
    # 2. Reconstruct PRPD
    print("Reconstructing PRPD (Kuramoto + Frequency Optimization)...")
    phases, peaks_out = reconstruct_blind_prpd(toa_s, peaks, freq_hz=50.0, auto_calibrate=True)
    
    # 3. Plotting with Amplitude-Weighted KDE
    y_raw_all = peaks_out
    x_all = phases
    y_all = np.where((x_all >= 0) & (x_all <= 180), y_raw_all, -y_raw_all)
    
    # Outlier detection
    theta2 = np.deg2rad(x_all) * 2
    mean_angle = np.arctan2(np.mean(np.sin(theta2)), np.mean(np.cos(theta2))) / 2.0
    center1 = np.rad2deg(mean_angle) % 360
    center2 = (center1 + 180) % 360
    d1 = np.minimum(np.abs(x_all - center1), 360 - np.abs(x_all - center1))
    d2 = np.minimum(np.abs(x_all - center2), 360 - np.abs(x_all - center2))
    d_min = np.minimum(d1, d2)
    median_d = np.median(d_min)
    mad = np.median(np.abs(d_min - median_d))
    threshold = median_d + 3.0 * max(mad * 1.4826, 5.0)
    inlier_mask = d_min <= threshold
    
    x_inliers = x_all[inlier_mask]
    y_inliers = y_all[inlier_mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gray Outliers
    if np.sum(~inlier_mask) > 0:
        ax.scatter(x_all[~inlier_mask], y_all[~inlier_mask], c="lightgray", s=6, alpha=0.4, edgecolors="none", zorder=1)
        
    print("Computing Amplitude-Weighted KDE...")
    if len(x_inliers) > 5000:
        idx = np.random.choice(np.arange(len(x_inliers)), 5000, replace=False)
        xy_sample = np.vstack([x_inliers[idx], y_inliers[idx]])
        w_sample = np.abs(y_inliers[idx])
    else:
        xy_sample = np.vstack([x_inliers, y_inliers])
        w_sample = np.abs(y_inliers)
        
    try:
        kde = gaussian_kde(xy_sample, bw_method=0.15, weights=w_sample)
        z = kde(np.vstack([x_inliers, y_inliers]))
        idx_sort = z.argsort()
        xp, yp, zp = x_inliers[idx_sort], y_inliers[idx_sort], z[idx_sort]
        sizes = 8 + 35 * (np.abs(yp) / np.max(np.abs(yp)))
    except Exception as e:
        print(f"KDE failed: {e}")
        xp, yp = x_inliers, y_inliers
        zp, sizes = "blue", 12
        
    scatter = ax.scatter(xp, yp, c=zp, cmap="turbo", s=sizes, alpha=0.9, edgecolors="none", zorder=2)
    fig.colorbar(scatter, ax=ax, label="Densidad Relativa (Ponderada por V)")
    
    # Reference Sine
    t_sin = np.linspace(0, 360, 360)
    max_amp = np.max(np.abs(y_all)) * 1.05
    ax.plot(t_sin, max_amp * np.sin(np.radians(t_sin)), color="red", alpha=0.4, linewidth=1.5, zorder=0)
    
    ax.set_title(f"Blind PRPD (Amplitude-Weighted) - {mat_file.name}")
    ax.set_xlabel("Fase (grados)")
    ax.set_ylabel("Amplitud del Pulso")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.grid(True, linestyle=":", alpha=0.4)
    
    out_file = out_dir / f"prpd_mat_{mat_file.stem}.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    plot_mat_prpd()
