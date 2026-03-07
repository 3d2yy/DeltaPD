import numpy as np
from scipy.stats import weibull_min

def fit_weibull_moving(delta_t: np.ndarray, window: int = 50, min_periods: int = 10):
    """
    Ajusta una distribución Weibull sobre ventanas móviles de inter-pulse times.
    
    Parameters
    ----------
    delta_t : np.ndarray
        Vector de Tiempos de espera entre pulsos.
    window : int
        Tamaño de la ventana rodante en cantidad de eventos.
    min_periods : int
        Mínimo de eventos requeridos para no devolver NaN.
        
    Returns
    -------
    shape_beta : np.ndarray
        Parámetro de forma (beta). beta < 1: decreciente, beta=1: Poisson, beta > 1: avalancha.
    scale_eta : np.ndarray
        Parámetro de escala (eta).
    """
    n = len(delta_t)
    out_beta = np.full(n, np.nan)
    out_eta = np.full(n, np.nan)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        chunk = delta_t[start_idx:i+1]
        chunk = chunk[chunk > 0]
        if len(chunk) >= min_periods:
            try:
                # floc=0 fuerza el inicio en 0 (standar Weibull de dos parámetros)
                params = weibull_min.fit(chunk, floc=0)
                out_beta[i] = params[0] # Forma
                out_eta[i] = params[2]  # Escala
            except Exception:
                pass
                
    return out_beta, out_eta

def compute_burstiness_index(delta_t: np.ndarray, window: int = 50, min_periods: int = 10):
    """
    Calcula el Burstiness Index (B) sobre ventanas móviles.
    B = (sigma - mu) / (sigma + mu)
    B = 1: Señal altamente en ráfagas (Burst)
    B = 0: Proceso de Poisson (Random)
    B = -1: Señal periódica regular (Regular)
    """
    n = len(delta_t)
    out_burstiness = np.full(n, np.nan)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        chunk = delta_t[start_idx:i+1]
        if len(chunk) >= min_periods:
            mu = np.mean(chunk)
            sigma = np.std(chunk)
            if (sigma + mu) > 0:
                b = (sigma - mu) / (sigma + mu)
                out_burstiness[i] = b
                
    return out_burstiness
