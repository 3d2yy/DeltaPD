import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import scipy.signal

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.blind_prpd import reconstruct_blind_prpd

def plot_objective():
    folder = "Prueba 1 - Internas"
    channel = "CH3"
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    file_path = base_dir / folder / f"{channel}.csv"
    
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    f_test = np.linspace(49.8, 50.2, 50000)
    R2 = []
    
    for f in f_test:
        z2 = np.exp(1j * 4 * np.pi * f * toa_s)
        R2.append(np.abs(np.mean(z2)))
        
    R2 = np.array(R2)
    
    # Penalize far from 50.0 with Gaussian sigma = 0.03 Hz
    sigma = 0.02
    penalty = np.exp(-((f_test - 50.0)**2) / (2 * sigma**2))
    R2_penalized = R2 * penalty
    
    plt.figure(figsize=(10, 6))
    plt.plot(f_test, R2, label="Raw R(f)", alpha=0.5)
    plt.plot(f_test, R2_penalized, label="Penalized R(f)", alpha=0.9, color='red')
    plt.axvline(50.0, color='k', linestyle='--', label="50.0 Hz nominal")
    plt.title("Kuramoto Parameter vs Frequency (Prueba 1) - Aliasing Fix")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("R (Concentration)")
    plt.legend()
    plt.grid()
    plt.savefig(base_dir / "DeltaPD_improved" / "outputs" / "R_freq_p1_penalized.png")
    plt.close()
    
    best_f2 = f_test[np.argmax(R2)]
    best_f2_penalized = f_test[np.argmax(R2_penalized)]
    print(f"Best f raw: {best_f2:.4f} Hz")
    print(f"Best f penalized: {best_f2_penalized:.4f} Hz")
    
    def plot_prpd(f, fname):
        phases_deg, _ = reconstruct_blind_prpd(toa_s, peaks, freq_hz=f, auto_calibrate=False)
        plt.figure()
        plt.scatter(phases_deg, peaks, alpha=0.5, s=10, c='purple')
        plt.xlim(0, 360)
        plt.title(f"PRPD at f={f:.4f} Hz")
        plt.xlabel("Phase (deg)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig(base_dir / "DeltaPD_improved" / "outputs" / fname)
        plt.close()
        
    plot_prpd(50.0, "prpd_p1_50_00.png")
    plot_prpd(best_f2, "prpd_p1_best_raw.png")
    plot_prpd(best_f2_penalized, "prpd_p1_best_penalized.png")

if __name__ == '__main__':
    plot_objective()
