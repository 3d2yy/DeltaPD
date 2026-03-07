import numpy as np

def calibrate_grid_frequency(toa_s: np.ndarray, base_freq: float = 50.0, search_width: float = 0.5, steps: int = 200000) -> float:
    """
    Calibra la frecuencia efectiva de la red eléctrica relativa al reloj del DAQ.
    Usa búsqueda en dos etapas para máxima precisión:
      1. Barrido grueso sobre [base-width, base+width]
      2. Refinamiento ultrafino alrededor del mejor candidato (±0.01 Hz, 100k pasos)
    
    Compensa automáticamente el clock-skew del oscilador de cuarzo del osciloscopio.
    """
    if len(toa_s) < 10:
        return base_freq
    
    toa_work = toa_s
    if len(toa_s) > 10000:
        toa_work = np.random.choice(toa_s, 10000, replace=False)
        
    # --- Etapa 1: Barrido grueso ---
    f_test = np.linspace(base_freq - search_width, base_freq + search_width, steps)
    phase_matrix = 4 * np.pi * np.outer(f_test, toa_work)
    z = np.exp(1j * phase_matrix)
    R = np.abs(np.mean(z, axis=1))
    coarse_best = f_test[np.argmax(R)]
    
    # --- Etapa 2: Refinamiento ultrafino ---
    f_fine = np.linspace(coarse_best - 0.01, coarse_best + 0.01, 100000)
    phase_fine = 4 * np.pi * np.outer(f_fine, toa_work)
    z_fine = np.exp(1j * phase_fine)
    R_fine = np.abs(np.mean(z_fine, axis=1))
    best_freq = f_fine[np.argmax(R_fine)]
    
    return float(best_freq)

def _remove_phase_outliers(phases_deg: np.ndarray, peaks: np.ndarray, sigma_threshold: float = 2.5):
    """
    Elimina eventos que caen fuera de los dos clusters principales del PRPD.
    Usa la distancia circular al centroide más cercano como criterio.
    """
    if len(phases_deg) < 20:
        return phases_deg, peaks
    
    # Encontrar los dos centroides usando el doble-ángulo
    theta2 = np.deg2rad(phases_deg) * 2
    mean_angle = np.arctan2(np.mean(np.sin(theta2)), np.mean(np.cos(theta2))) / 2.0
    center1 = np.rad2deg(mean_angle) % 360
    center2 = (center1 + 180) % 360
    
    # Distancia circular al centroide más cercano
    d1 = np.minimum(np.abs(phases_deg - center1), 360 - np.abs(phases_deg - center1))
    d2 = np.minimum(np.abs(phases_deg - center2), 360 - np.abs(phases_deg - center2))
    d_min = np.minimum(d1, d2)
    
    # Umbral estadístico basado en la dispersión circular del cluster
    median_d = np.median(d_min)
    mad = np.median(np.abs(d_min - median_d))
    threshold = median_d + sigma_threshold * max(mad * 1.4826, 5.0)  # mínimo 5° de tolerancia
    
    mask = d_min <= threshold
    n_removed = np.sum(~mask)
    if n_removed > 0:
        print(f"[blind_prpd] Filtrados {n_removed} outliers de fase (umbral={threshold:.1f}°)")
    
    return phases_deg[mask], peaks[mask]

def reconstruct_blind_prpd(toa_s: np.ndarray, peaks: np.ndarray, freq_hz: float = 50.0, auto_calibrate: bool = True):
    """
    Reconstruye el patrón PRPD ciegamente (sin medir tensión de la red AC).
    Incluye calibración de frecuencia de dos etapas y filtrado de outliers.
    """
    if len(toa_s) == 0:
        return np.array([]), np.array([])
        
    if auto_calibrate:
        best_freq = calibrate_grid_frequency(toa_s, base_freq=freq_hz)
        print(f"[{__name__}] Frecuencia ciega calibrada: {best_freq:.6f} Hz (Base: {freq_hz} Hz)")
        freq_to_use = best_freq
    else:
        freq_to_use = freq_hz
        
    T_s = 1.0 / freq_to_use
    t_mod = np.mod(toa_s, T_s)
    phases_deg = (t_mod / T_s) * 360.0
    
    # Centrado automático: cluster principal en ~70°
    theta = np.deg2rad(phases_deg) * 2
    avg_theta = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))) / 2.0
    avg_deg = np.rad2deg(avg_theta)
    
    target_deg = 70.0 
    shift_deg = target_deg - avg_deg
    phases_deg = np.mod(phases_deg + shift_deg, 360.0)
    
    return phases_deg, peaks

