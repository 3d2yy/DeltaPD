import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses

def analyze_drift():
    folder = "Prueba 1 - Internas"
    channel = "CH3"
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    file_path = base_dir / folder / f"{channel}.csv"
    
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    # Calculate raw phase based on exactly 50 Hz
    t_mod = np.mod(toa_s, 1.0 / 50.0)
    raw_phase_deg = (t_mod / (1.0 / 50.0)) * 360.0
    
    # Let's project modulo 180 degrees since PD is symmetric
    half_cycles = np.floor(toa_s * 100.0)  # number of half cycles (20 ms / 2 = 10 ms = 100 Hz grid)
    p_raw = toa_s * 50.0
    p_frac = np.mod(p_raw, 0.5) * 360 * 2  # scale to 0-360 for half cycle
    
    plt.figure(figsize=(10, 6))
    plt.scatter(toa_s, p_frac, alpha=0.5, s=10)
    plt.title("Raw Phase Drift Over Time (modulo 180 deg scaled to 360)")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase in Half-Cycle (deg)")
    plt.grid()
    plt.savefig(base_dir / "DeltaPD_improved" / "outputs" / "phase_drift_p1.png")
    plt.close()
    
    # If we unwrap the phase of the clusters?
    # We want to find the drift curve phi(t).
    
if __name__ == '__main__':
    analyze_drift()
