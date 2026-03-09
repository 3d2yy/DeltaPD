import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import scipy.interpolate
import scipy.integrate

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.blind_prpd import calibrate_grid_frequency

def test_adaptive():
    folder = "Prueba 1 - Internas"
    channel = "CH3"
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    file_path = base_dir / folder / f"{channel}.csv"
    
    x, fs, times_abs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x, fs, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_s = times_abs[pulse_indices]
    peaks = np.abs(x[pulse_indices])
    
    t_min = toa_s[0]
    t_max = toa_s[-1]
    
    window_size = 10.0 # seconds
    step = 2.0 # seconds
    
    t_nodes = np.arange(t_min, t_max + step, step)
    f_nodes = []
    t_actual_nodes = []
    
    print("Computing local frequencies...")
    for tc in t_nodes:
        mask = (toa_s >= tc - window_size/2) & (toa_s < tc + window_size/2)
        local_toa = toa_s[mask]
        
        if len(local_toa) > 10:
            f_local = calibrate_grid_frequency(local_toa, base_freq=50.0, search_width=0.05, steps=10000)
            f_nodes.append(f_local)
            t_actual_nodes.append(tc)
            
    f_nodes = np.array(f_nodes)
    t_actual_nodes = np.array(t_actual_nodes)
    
    # Smooth local frequencies
    from scipy.signal import savgol_filter
    if len(f_nodes) > 5:
        f_nodes_smooth = savgol_filter(f_nodes, window_length=5, polyorder=2)
    else:
        f_nodes_smooth = f_nodes
        
    print(f"Frequencies min: {f_nodes_smooth.min():.4f}, max: {f_nodes_smooth.max():.4f}")
    
    # Interpolate f(t)
    f_interp = scipy.interpolate.interp1d(t_actual_nodes, f_nodes_smooth, kind='linear', fill_value="extrapolate")
    
    # Integrate to get phase Phase(t) = integral(f(t) dt)
    # create a dense grid
    t_dense = np.linspace(t_min, t_max, 50000)
    f_dense = f_interp(t_dense)
    phase_dense = scipy.integrate.cumulative_trapezoid(f_dense, t_dense, initial=0.0)
    
    # get exact phase for each toa
    phase_interp = scipy.interpolate.interp1d(t_dense, phase_dense, kind='linear')
    phase_events = phase_interp(toa_s)
    
    # Base phase offset - shift it so the max density is at 45/225
    phase_deg = np.mod(phase_events, 1.0) * 360.0
    
    plt.figure()
    plt.scatter(phase_deg, peaks, alpha=0.5, s=10, c='red')
    plt.xlim(0, 360)
    plt.title("Adaptive PRPD (Time-varying Grid Frequency)")
    plt.xlabel("Phase (deg)")
    plt.ylabel("Amplitude (V)")
    plt.grid()
    plt.savefig(base_dir / "DeltaPD_improved" / "outputs" / "adaptive_prpd_p1.png")
    plt.close()
    
    plt.figure()
    plt.plot(t_actual_nodes, f_nodes, 'o', label='Local f (raw)')
    plt.plot(t_actual_nodes, f_nodes_smooth, '-', label='Local f (smoothed)')
    plt.title("Grid Frequency over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid()
    plt.savefig(base_dir / "DeltaPD_improved" / "outputs" / "f_drift_over_time.png")
    plt.close()

if __name__ == '__main__':
    test_adaptive()
