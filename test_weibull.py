import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'e:/Carpeta definitiva de Tesis/programas/DeltaPD-main/src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.statistics import fit_weibull_moving, compute_burstiness_index

def test_weibull():
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    folder = "Prueba 1 - Internas"
    channel = "CH3"
    file_path = base_dir / folder / f"{channel}.csv"
    
    print("Loading data...")
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=10e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    
    # Calculate delta_t
    delta_t = np.diff(toa_s)
    
    print(f"Total pulses: {len(toa_s)}, Total delta_t: {len(delta_t)}")
    
    # Compute Weibull and Burstiness
    window = 100
    print(f"Computing Weibull with window {window}...")
    beta, eta = fit_weibull_moving(delta_t, window=window, min_periods=window//2)
    
    print("Computing Burstiness Index...")
    burstiness = compute_burstiness_index(delta_t, window=window, min_periods=window//2)
    
    # Simple plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    t_plot = toa_s[1:] # Align with delta_t
    
    axes[0].scatter(t_plot, delta_t, s=2, alpha=0.5, c='gray')
    axes[0].set_ylabel("$\Delta t$ (s)")
    axes[0].set_yscale('log')
    axes[0].set_title("Inter-pulse times")
    
    axes[1].plot(t_plot, beta, c='blue', alpha=0.7)
    axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel("Weibull $\\beta$")
    axes[1].set_ylim(0, 3)
    
    axes[2].plot(t_plot, burstiness, c='green', alpha=0.7)
    axes[2].set_ylabel("Burstiness Index $B$")
    axes[2].set_ylim(-1, 1)
    axes[2].axhline(0.0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel("Time (s)")
    
    plt.tight_layout()
    out = "e:/Carpeta definitiva de Tesis/programas/DeltaPD_improved/outputs/test_weibull.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot to {out}")
    
    # Print some stats
    print(f"Valid Beta values: {np.sum(~np.isnan(beta))}")
    print(f"Mean Beta: {np.nanmean(beta):.3f}")
    print(f"Mean Burstiness: {np.nanmean(burstiness):.3f}")

if __name__ == "__main__":
    test_weibull()
