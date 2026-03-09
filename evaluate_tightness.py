import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.blind_prpd import calibrate_grid_frequency, reconstruct_blind_prpd

def evaluate_tightness():
    folder = "Prueba 1 - Internas"
    channel = "CH3"
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    file_path = base_dir / folder / f"{channel}.csv"
    
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    def measure_variance(phases):
        theta = np.deg2rad(phases) * 2
        R = np.abs(np.mean(np.exp(1j * theta)))
        return 1 - R
        
    print("Evaluating Circular Variance (lower is tighter)...")
    
    p1 = np.mod(toa_s, 1.0/50.0) * 50.0 * 360.0
    print(f"50.0000 Hz: var = {measure_variance(p1):.4f}")
    
    f_test = np.linspace(49.8, 50.2, 50000)
    z = np.exp(1j * 4 * np.pi * np.outer(f_test, toa_s[:min(10000, len(toa_s))]))
    R_raw = np.abs(np.mean(z, axis=1))
    f_raw = f_test[np.argmax(R_raw)]
    p2 = np.mod(toa_s, 1.0/f_raw) * f_raw * 360.0
    print(f"Raw Kuramoto ({f_raw:.4f} Hz) [POTENTIAL ALIAS]: var = {measure_variance(p2):.4f}")
    
    penalty = np.exp(-((f_test - 50.0)**2) / (2 * 0.02**2))
    R_pen = R_raw * penalty
    f_pen = f_test[np.argmax(R_pen)]
    p3 = np.mod(toa_s, 1.0/f_pen) * f_pen * 360.0
    print(f"Penalized Kuramoto ({f_pen:.4f} Hz): var = {measure_variance(p3):.4f}")

if __name__ == '__main__':
    evaluate_tightness()
