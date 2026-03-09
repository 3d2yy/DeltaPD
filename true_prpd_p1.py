import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd
import scipy.signal

sys.path.insert(0, 'src')
from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.campaign.plot_material import plot_blind_prpd

def find_ac_channel(base_dir, folder):
    for ch in ["CH1", "CH2", "CH3", "CH4"]:
        file_path = base_dir / folder / f"{ch}.csv"
        if not file_path.exists():
            continue
        try:
            # Load only the first few segments to save time
            x, fs = load_empirical_signal(str(file_path), preserve_amplitude=True, include_absolute_times=False)
            rms = np.sqrt(np.mean(x**2))
            peak = np.max(np.abs(x))
            crest_factor = peak / (rms + 1e-12)
            print(f"{ch}: Crest Factor = {crest_factor:.2f}")
            if crest_factor < 5.0:
                print(f"--> Found AC reference in {ch}")
                return file_path
        except Exception as e:
            print(f"Error {ch}: {e}")
    return None

def extract_true_phase(x_ac, fs_ac, toa_pd):
    # Denoise AC
    # Decimate heavily first since AC is 50Hz and fs is ~ 5GHz
    decimate_fs = 10000.0
    decimate_factor = int(fs_ac / decimate_fs)
    print(f"Decimation factor: {decimate_factor}")
    
    # We must operate directly on zero crossings of the raw wave using a simple smoothing
    window = int(fs_ac * 0.002) # 2ms window
    if window % 2 == 0: window += 1
    x_ac_filt = scipy.signal.savgol_filter(x_ac, window, 3)
    
    # Simple zero-crossing detection
    zcs = np.where(np.diff(np.sign(x_ac_filt)) > 0)[0]
    t_zcs = zcs / fs_ac
    print(f"Found {len(t_zcs)} positive zero crossings.")
    
    phases = []
    for t in toa_pd:
        idx = np.searchsorted(t_zcs, t) - 1
        if idx < 0 or idx >= len(t_zcs) - 1:
            phases.append(np.nan)
        else:
            t_start = t_zcs[idx]
            t_end = t_zcs[idx+1]
            period = t_end - t_start
            phase = ((t - t_start) / period) * 360.0
            phases.append(phase)
            
    return np.array(phases)

def main():
    folder = "Prueba 1 - Internas"
    base_dir = Path("e:/Carpeta definitiva de Tesis/programas")
    
    print("Loading PD data from CH3...")
    file_pd = base_dir / folder / "CH3.csv"
    x_pd, fs_pd, times_abs_pd = load_empirical_signal(str(file_pd), preserve_amplitude=True, include_absolute_times=True)
    pulse_indices = detect_pulses(x_pd, fs_pd, threshold_sigma=5.0, min_separation_s=20e-9, method="threshold")
    toa_pd = times_abs_pd[pulse_indices]
    peaks_pd = np.abs(x_pd[pulse_indices])
    
    ac_file = find_ac_channel(base_dir, folder)
    if ac_file:
        print(f"Loading AC reference from {ac_file}...")
        x_ac, fs_ac, times_abs_ac = load_empirical_signal(str(ac_file), preserve_amplitude=True, include_absolute_times=True)
        
        print("Extracting true phases...")
        # Since times_abs_ac has gaps, we need to extract phase separately per segment
        # Actually times_abs_ac provides absolute t for each sample
        phases = extract_true_phase(x_ac, fs_ac, toa_pd)
        
        valid = ~np.isnan(phases)
        phases_true = phases[valid]
        peaks_valid = peaks_pd[valid]
        
        print(f"Extracted {len(phases_true)} valid true phases.")
        
        df = pd.DataFrame({"prpd_phase_deg": phases_true, "peak_v": peaks_valid})
        out_png = str(base_dir / "DeltaPD_improved" / "outputs" / "08_true_prpd_internas.png")
        plot_blind_prpd(df, out_png)
        print(f"Generated Ground Truth PRPD at {out_png}")
    else:
        print("No AC channel found. Try CH2.")

if __name__ == '__main__':
    main()
