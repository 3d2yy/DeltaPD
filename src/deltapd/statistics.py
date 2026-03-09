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

def compute_fano_factor(toa_s: np.ndarray, bin_duration_s: float = 0.1, window_bins: int = 20, min_bins: int = 5):
    """
    Calcula el Factor de Fano (F) sobre ventanas temporales.
    F = Var(N) / Mean(N), donde N es el conteo de pulsos por bin temporal.
    
    F = 1: Proceso Poisson puro (aleatorio)
    F > 1: Sobre-dispersión (agrupamiento / clustering)
    F < 1: Sub-dispersión (regularidad / anti-clustering)
    """
    if len(toa_s) < 10:
        return np.array([]), np.array([])
    
    t_min, t_max = toa_s.min(), toa_s.max()
    bin_edges = np.arange(t_min, t_max + bin_duration_s, bin_duration_s)
    counts, _ = np.histogram(toa_s, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    n = len(counts)
    fano_values = np.full(n, np.nan)
    
    for i in range(n):
        start_idx = max(0, i - window_bins + 1)
        chunk = counts[start_idx:i+1]
        if len(chunk) >= min_bins and np.mean(chunk) > 0:
            fano_values[i] = np.var(chunk) / np.mean(chunk)
    
    return bin_centers, fano_values

def compute_phase_entropy(phases_deg: np.ndarray, window: int = 100, n_bins: int = 36, min_periods: int = 30):
    """
    Calcula la Entropía de Shannon normalizada sobre la distribución de fases PRPD.
    H_norm = H / log2(n_bins)
    """
    n = len(phases_deg)
    out_entropy = np.full(n, np.nan)
    h_max = np.log2(n_bins)
    
    bin_edges = np.linspace(0, 360, n_bins + 1)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        chunk = phases_deg[start_idx:i+1]
        if len(chunk) >= min_periods:
            hist, _ = np.histogram(chunk, bins=bin_edges)
            p = hist / hist.sum()
            p = p[p > 0]
            H = -np.sum(p * np.log2(p)) if len(p) > 0 else 0
            out_entropy[i] = H / h_max
    
    return out_entropy
