import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.campaign.plot_material import plot_blind_prpd

def test_methods():
    folder = "Prueba 1 - Internas"
    channel = "CH3"
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    file_path = base_dir / folder / f"{channel}.csv"
    
    print("Loading data...")
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    out_dir = base_dir / "DeltaPD_improved" / "outputs" / "debug_prpd"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def make_df(phases):
        return pd.DataFrame({"prpd_phase_deg": phases, "peak_v": peaks})
        
    print("1. Exactly 50.0 Hz")
    p1 = np.mod(toa_s, 1.0/50.0) * 50.0 * 360.0
    plot_blind_prpd(make_df(p1), str(out_dir / "method1_50hz.png"))
    
    print("2. Global Kuramoto (no penalty) - e.g. 49.89 Hz")
    from deltapd.blind_prpd import calibrate_grid_frequency
    f2 = calibrate_grid_frequency(toa_s, base_freq=50.0, search_width=0.2, steps=50000)
    print(f"  f2 = {f2}")
    p2 = np.mod(toa_s, 1.0/f2) * f2 * 360.0
    plot_blind_prpd(make_df(p2), str(out_dir / "method2_kuramoto_unpenalized.png"))
    
    print("3. Global Kuramoto (penalized 0.02 sigma)")
    # Re-implement penalized here just to be sure
    f_test = np.linspace(49.8, 50.2, 50000)
    z = np.exp(1j * 4 * np.pi * np.outer(f_test, toa_s[:min(10000, len(toa_s))]))
    R = np.abs(np.mean(z, axis=1))
    penalty = np.exp(-((f_test - 50.0)**2) / (2 * 0.02**2))
    R_pen = R * penalty
    f3 = f_test[np.argmax(R_pen)]
    print(f"  f3 = {f3}")
    p3 = np.mod(toa_s, 1.0/f3) * f3 * 360.0
    plot_blind_prpd(make_df(p3), str(out_dir / "method3_kuramoto_penalized.png"))
    
    print("4. Adaptive method")
    from deltapd.blind_prpd import reconstruct_adaptive_blind_prpd
    p4, _ = reconstruct_adaptive_blind_prpd(toa_s, peaks, base_freq=50.0, window_s=10.0, step_s=2.0)
    plot_blind_prpd(make_df(p4), str(out_dir / "method4_adaptive.png"))
    
    print("Done generating debug plots.")

if __name__ == '__main__':
    test_methods()
