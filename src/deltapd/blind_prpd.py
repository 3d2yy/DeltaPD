import numpy as np

def reconstruct_blind_prpd(toa_s: np.ndarray, peaks: np.ndarray, freq_hz: float = 50.0):
    """
    Reconstruye el patrón PRPD (Phase-Resolved Partial Discharge) ciegamente
    usando únicamente los tiempos de llegada absolutos (TOA) de los pulsos UHF.
    
    Parameters
    ----------
    toa_s : np.ndarray
        Vector con los tiempos de llegada de los pulsos en segundos.
        Debe ser un tiempo continuo o un tiempo ya corregido con los segmentos.
    peaks : np.ndarray
        Amplitudes máximas (peak voltage) correspondientes a cada pulso.
    freq_hz : float
        Frecuencia base de la red (típicamente 50.0 o 60.0 Hz).
        
    Returns
    -------
    phases_deg : np.ndarray
        Fase reconstruida en grados, rango [0, 360).
    peaks_aligned : np.ndarray
        Idéntico a `peaks`, por si se desea filtrar en el futuro.
    """
    if len(toa_s) == 0:
        return np.array([]), np.array([])
        
    T_s = 1.0 / freq_hz
    # Calculamos el módulo del tiempo respecto al período de onda iterativa
    t_mod = np.mod(toa_s, T_s)
    
    # Convertimos la fracción del período a grados eléctricos
    phases_deg = (t_mod / T_s) * 360.0
    
    return phases_deg, peaks
