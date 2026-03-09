import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.blind_prpd import calibrate_grid_frequency

def analyze_p1():
    folder = "Prueba 1 - Internas"
    channel = "CH3"
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    file_path = base_dir / folder / f"{channel}.csv"
    
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    print(f"Total time span: {toa_s[-1] - toa_s[0]:.2f} seconds")
    print(f"Total pulses: {len(toa_s)}")
    
    # Calculate gaps
    dt = np.diff(toa_s)
    large_gaps = dt[dt > 1.0]
    print(f"Number of gaps > 1s: {len(large_gaps)}")
    if len(large_gaps) > 0:
        print(f"Average large gap: {np.mean(large_gaps):.2f} s")
        print(f"Max gap: {np.max(large_gaps):.2f} s")

    # What if we calibrate in chunks?
    chunk_size = len(toa_s) // 5
    if chunk_size == 0: chunk_size = len(toa_s)
    
    for i in range(5):
        start_idx = i * chunk_size
        end_idx = min((i+1) * chunk_size, len(toa_s))
        chunk_toa = toa_s[start_idx:end_idx]
        if len(chunk_toa) > 10:
            f_cal = calibrate_grid_frequency(chunk_toa)
            print(f"Chunk {i+1} f_cal: {f_cal:.4f} Hz (span: {chunk_toa[-1] - chunk_toa[0]:.2f} s)")

if __name__ == '__main__':
    analyze_p1()
