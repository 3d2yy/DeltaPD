"""
Módulo de preprocesamiento de señales UHF para detección de descargas parciales.

Este módulo proporciona funciones para:
- Filtrado pasabanda
- Normalización de señales
- Extracción de envolvente mediante transformada de Hilbert
- Eliminación de ruido mediante wavelets
- Optimización estocástica de parámetros wavelet (Monte Carlo + Grid Search)

Phase 1 — Stochastic Optimization:
    Implements a grid search across wavelet families {db4, sym8, coif3} and
    thresholding rules {soft, hard} × {universal, minimax, sqtwolog}.  A Monte
    Carlo simulation (N=1000) injects AWGN into a reference UHF-PD signal and
    selects the configuration that minimises E[RMSE] subject to Var[RMSE] < ε.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy import signal
from scipy.signal import hilbert

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Signal = NDArray[np.floating[Any]]


# ===================================================================
# Phase 1 — Stochastic wavelet optimisation data structures
# ===================================================================


@dataclass
class WaveletGridPoint:
    """Single point in the wavelet parameter grid."""

    wavelet: str
    threshold_mode: str  # 'soft' | 'hard'
    threshold_rule: str  # 'universal' | 'minimax' | 'sqtwolog'
    rmse_mean: float = 0.0
    rmse_var: float = 0.0
    rmse_samples: List[float] = field(default_factory=list)


@dataclass
class MonteCarloResult:
    """Aggregate result of the Monte-Carlo grid search."""

    best_wavelet: str
    best_threshold_mode: str
    best_threshold_rule: str
    best_rmse_mean: float
    best_rmse_var: float
    grid: List[WaveletGridPoint] = field(default_factory=list)
    n_iterations: int = 1000
    epsilon: float = 1e-3
    n_feasible: int = 0
    feasibility_rate: float = 0.0
    converged: bool = True


# ===================================================================
# Helper — reference UHF-PD signal generation
# ===================================================================


def generate_uhf_pd_signal_physical(
    n_samples: int = 4096,
    fs: float = 1e9,
    n_pulses: int = 12,
    snr_db: float = 30.0,
    seed: Optional[int] = None,
    # --- Dielectric channel parameters ---
    epsilon_r: float = 2.2,
    tan_delta: float = 0.005,
    propagation_distance_m: float = 0.3,
    # --- Vivaldi antenna parameters ---
    f_low_hz: float = 300e6,
    f_high_hz: float = 3e9,
    antenna_order: int = 4,
    # --- PD current-pulse parameters ---
    tau1_range_ns: Tuple[float, float] = (0.5, 2.0),
    tau2_range_ns: Tuple[float, float] = (5.0, 20.0),
    amplitude_range: Tuple[float, float] = (0.5, 2.0),
    # --- Environmental noise parameters ---
    nbi_amplitudes: Sequence[float] = (0.05, 0.02),
    nbi_frequencies_hz: Sequence[float] = (900e6, 1.8e9),
    corona_lambda: float = 200.0,
) -> Tuple[Signal, Signal]:
    """Generate a physics-based UHF partial-discharge signal.

    The model chains physically-motivated stages:
    1. **PD current pulse** — Gemant-Philippoff double-exponential.
    2. **Dielectric channel transfer function** — Complex permittivity `ε*(f)`.
    3. **Antipodal Vivaldi antenna** — Butterworth bandpass.
    4. **Narrowband Interference (NBI)** — CW sinusoids (e.g. GSM / Broadcast).
    5. **Corona Noise** — Stochastic high-rate, low-amplitude damped sinusoids.
    """
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Stage 1 — Generate PD pulses
    current_signal: Signal = np.zeros(n_samples, dtype=np.float64)
    for _ in range(n_pulses):
        pos = rng.integers(50, n_samples - 200)
        tau1 = rng.uniform(*tau1_range_ns) * 1e-9
        tau2 = rng.uniform(*tau2_range_ns) * 1e-9
        amp = rng.uniform(*amplitude_range)
        dur = min(int(50e-9 * fs), n_samples - pos)
        t_local = np.arange(dur) / fs
        pulse = amp * (np.exp(-t_local / tau2) - np.exp(-t_local / tau1))
        current_signal[pos : pos + dur] += pulse

    # Stage 2 — Dielectric channel
    eps_0 = 8.854187817e-12
    mu_0 = 4.0 * np.pi * 1e-7
    d = propagation_distance_m
    eps_complex = epsilon_r * eps_0 * (1.0 - 1j * tan_delta)

    freqs_safe = freqs.copy()
    freqs_safe[0] = 1.0
    omega = 2.0 * np.pi * freqs_safe
    k = omega * np.sqrt(mu_0 * eps_complex)
    H_dielectric = np.exp(-1j * k * d)
    H_dielectric[0] = 0.0

    # Stage 3 — Vivaldi antenna
    f_center = np.sqrt(f_low_hz * f_high_hz)
    bw = f_high_hz - f_low_hz
    H_ant_mag2 = 1.0 / (
        1.0 + ((freqs_safe**2 - f_center**2) / (freqs_safe * bw)) ** (2 * antenna_order)
    )
    H_antenna = np.sqrt(H_ant_mag2)
    H_antenna[0] = 0.0

    I_f = np.fft.rfft(current_signal)
    V_f = I_f * H_dielectric * H_antenna
    clean: Signal = np.fft.irfft(V_f, n=n_samples)

    # Stages 4 & 5 — AWGN, NBI, and Corona Noise
    sig_power = np.mean(clean**2) + 1e-30
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    awgn = rng.normal(0.0, np.sqrt(noise_power), n_samples)

    t_full = np.arange(n_samples) / fs
    nbi = np.zeros_like(t_full)
    for amp_ratio, f_nbi in zip(nbi_amplitudes, nbi_frequencies_hz):
        nbi += (amp_ratio * np.max(clean)) * np.sin(2.0 * np.pi * f_nbi * t_full)

    corona = np.zeros_like(t_full)
    n_corona_pulses = rng.poisson(corona_lambda * (n_samples / fs))
    for _ in range(n_corona_pulses):
        pos_c = rng.integers(0, n_samples - 100)
        f_c = rng.uniform(20e6, 80e6)
        tau_c = rng.uniform(5e-9, 20e-9)
        amp_c = rng.uniform(0.01, 0.05) * np.max(clean)
        dur_c = min(100, n_samples - pos_c)
        t_c = np.arange(dur_c) / fs
        corona[pos_c : pos_c + dur_c] += (
            amp_c * np.exp(-t_c / tau_c) * np.sin(2.0 * np.pi * f_c * t_c)
        )

    noisy: Signal = clean + awgn + nbi + corona
    return clean, noisy


# Backward-compatible alias
generate_uhf_reference_signal = generate_uhf_pd_signal_physical


def compute_rmse(reference: Signal, estimate: Signal) -> float:
    """Root Mean Square Error between *reference* and *estimate*."""
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


# ===================================================================
# Helper — wavelet threshold value by rule
# ===================================================================


def _sure_threshold(c: NDArray[np.floating[Any]]) -> float:
    r"""Stein's Unbiased Risk Estimator (SURE) threshold formulation.

    Let :math:`\mathbf{x} \in \mathbb{R}^d` be a coefficient vector with additive Gaussian noise
    where :math:`x_i \sim \mathcal{N}(\mu_i, \sigma^2)`.
    SURE provides an unbiased estimate for the quadratic risk under soft thresholding by :math:`t`:

    .. math::
        SURE(t; \mathbf{x}) = d + \sum_{i=1}^d \min(x_i^2, t^2) - 2 \cdot |\{i : |x_i| \le t\}|

    The optimal threshold minimizes this risk: :math:`t^* = \arg\min_{t \ge 0} SURE(t; \mathbf{x})`.
    """
    n = len(c)
    if n == 0:
        return 0.0
    c_sort = np.sort(np.abs(c)) ** 2
    c_cumsum = np.cumsum(c_sort)
    # Vectorized computation of SURE risk for each threshold point
    risk = n - 2 * np.arange(1, n + 1) + c_cumsum + c_sort * np.arange(n - 1, -1, -1)
    return float(np.sqrt(c_sort[np.argmin(risk)]))


def _threshold_value(
    coeffs: List[NDArray[np.floating[Any]]],
    n: int,
    rule: str,
) -> float:
    r"""Compute a wavelet threshold value according to *rule*.

    Universal (Donoho-Johnstone) formula:
    .. math:: \lambda_{univ} = \sigma \sqrt{2 \ln(N)}

    Where :math:`\sigma` is estimated via Median Absolute Deviation (MAD):
    .. math:: \sigma = \frac{\text{median}(|cD_1|)}{0.6745}
    """
    sigma: float = float(np.median(np.abs(coeffs[-1]))) / 0.6745 + 1e-15

    if rule == "universal":
        return sigma * np.sqrt(2.0 * np.log(n))
    elif rule == "sure":
        c_flat = np.concatenate(coeffs[1:])
        t_sure = _sure_threshold(c_flat / sigma)
        return sigma * t_sure
    elif rule == "sqtwolog":
        return sigma * np.sqrt(2.0 * np.log(n))
    elif rule == "minimax":
        if n <= 32:
            return 0.0
        return sigma * (0.3936 + 0.1829 * np.log2(n))
    else:
        raise ValueError(f"Unknown threshold rule: {rule!r}")


def wavelet_denoise_parametric(
    signal_data: Signal,
    wavelet: str = "db4",
    threshold_mode: str = "hard",
    threshold_rule: str = "minimax",
    level: Optional[int] = None,
) -> Signal:
    """Denoise *signal_data* using the DWT with full parametric control.

    Parameters
    ----------
    signal_data : Signal
        Input (noisy) signal.
    wavelet : str
        Wavelet family identifier (e.g. ``'db4'``, ``'sym8'``, ``'coif3'``).
    threshold_mode : str
        ``'soft'`` or ``'hard'`` thresholding.
    threshold_rule : str
        ``'universal'``, ``'minimax'``, or ``'sqtwolog'``.
    level : int, optional
        Decomposition level.  ``None`` → automatic.

    Returns
    -------
    Signal
        Denoised signal.
    """
    data = np.asarray(signal_data, dtype=np.float64)
    n = len(data)

    if level is None:
        level = min(pywt.dwt_max_level(n, wavelet), 6)

    coeffs = pywt.wavedec(data, wavelet, level=level)
    lam = _threshold_value(coeffs, n, threshold_rule)

    coeffs_t = [coeffs[0]]  # keep approximation
    for c in coeffs[1:]:
        coeffs_t.append(pywt.threshold(c, lam, mode=threshold_mode))

    rec: Signal = pywt.waverec(coeffs_t, wavelet)
    # Trim / pad to original length
    if len(rec) > n:
        rec = rec[:n]
    elif len(rec) < n:
        rec = np.pad(rec, (0, n - len(rec)), mode="edge")
    return rec


# ===================================================================
# Phase 1 — Monte-Carlo grid search
# ===================================================================


def monte_carlo_wavelet_optimization(
    reference_clean: Optional[Signal] = None,
    reference_noisy: Optional[Signal] = None,
    fs: float = 1e9,
    wavelet_families: Sequence[str] = ("db4", "sym8", "coif3"),
    threshold_modes: Sequence[str] = ("soft", "hard"),
    threshold_rules: Sequence[str] = ("universal", "minimax", "sqtwolog"),
    n_iterations: int = 1000,
    snr_range_db: Tuple[float, float] = (5.0, 25.0),
    epsilon: float = 1e-3,
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> MonteCarloResult:
    """Stochastic optimisation of wavelet denoising parameters.

    Performs a full-factorial grid search over *wavelet_families* ×
    *threshold_modes* × *threshold_rules*.  For each grid point, **N**
    Monte-Carlo iterations inject AWGN at a random SNR drawn uniformly from
    ``snr_range_db`` into the reference signal and measure the RMSE of the
    reconstructed (denoised) signal against the clean reference.

    The selection criterion minimises **E[RMSE]** subject to
    **Var[RMSE] < ε**.

    Parameters
    ----------
    reference_clean : Signal, optional
        Clean reference UHF-PD signal.  If ``None`` a synthetic one is
        generated internally.
    reference_noisy : Signal, optional
        Ignored when *reference_clean* is given (noise is injected by MC).
    fs : float
        Sampling frequency in Hz.
    wavelet_families : sequence of str
        Wavelet identifiers to search.
    threshold_modes : sequence of str
        ``'soft'`` and/or ``'hard'``.
    threshold_rules : sequence of str
        ``'universal'``, ``'minimax'``, ``'sqtwolog'``.
    n_iterations : int
        Number of Monte-Carlo realisations per grid point.
    snr_range_db : tuple of float
        (min, max) SNR in dB for AWGN injection.
    epsilon : float
        Upper bound on acceptable RMSE variance.
    seed : int, optional
        RNG seed for reproducibility.
    verbose : bool
        Print progress information.

    Returns
    -------
    MonteCarloResult
        Aggregated result including the optimal configuration and the full
        grid of evaluated points.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Generate (or accept) a reference clean signal
    # ------------------------------------------------------------------
    if reference_clean is None:
        reference_clean, _ = generate_uhf_reference_signal(
            n_samples=4096, fs=fs, n_pulses=12, snr_db=40.0, seed=seed
        )

    n = len(reference_clean)
    sig_power: float = float(np.mean(reference_clean**2)) + 1e-30

    # ------------------------------------------------------------------
    # Build parameter grid
    # ------------------------------------------------------------------
    grid: List[WaveletGridPoint] = []
    combos = list(itertools.product(wavelet_families, threshold_modes, threshold_rules))

    for idx, (wv, tmode, trule) in enumerate(combos):
        gp = WaveletGridPoint(wavelet=wv, threshold_mode=tmode, threshold_rule=trule)

        if verbose:
            print(
                f"  [{idx + 1}/{len(combos)}] wavelet={wv}, "
                f"mode={tmode}, rule={trule} — running {n_iterations} MC iterations …"
            )

        rmse_samples: List[float] = []
        for _ in range(n_iterations):
            snr_db = rng.uniform(*snr_range_db)
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = rng.normal(0.0, np.sqrt(noise_power), n)
            noisy = reference_clean + noise

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                denoised = wavelet_denoise_parametric(
                    noisy, wavelet=wv, threshold_mode=tmode, threshold_rule=trule
                )

            rmse_samples.append(compute_rmse(reference_clean, denoised))

        gp.rmse_samples = rmse_samples
        gp.rmse_mean = float(np.mean(rmse_samples))
        gp.rmse_var = float(np.var(rmse_samples))
        grid.append(gp)

    # ------------------------------------------------------------------
    # Selection: minimise E[RMSE]  s.t.  Var[RMSE] < ε
    # ------------------------------------------------------------------
    feasible = [gp for gp in grid if gp.rmse_var < epsilon]
    n_feasible = len(feasible)
    converged = n_feasible > 0

    if not converged:
        # Relax: choose from all grid points sorted by variance, then RMSE
        feasible = sorted(grid, key=lambda g: (g.rmse_var, g.rmse_mean))

    best = min(feasible, key=lambda g: g.rmse_mean)

    result = MonteCarloResult(
        best_wavelet=best.wavelet,
        best_threshold_mode=best.threshold_mode,
        best_threshold_rule=best.threshold_rule,
        best_rmse_mean=best.rmse_mean,
        best_rmse_var=best.rmse_var,
        grid=grid,
        n_iterations=n_iterations,
        epsilon=epsilon,
        n_feasible=n_feasible,
        feasibility_rate=n_feasible / len(grid) if len(grid) > 0 else 0.0,
        converged=converged,
    )

    if verbose:
        tag = "OK converged" if converged else "⚠ relaxed (no feasible point under ε)"
        print(
            f"\n  Optimal: wavelet={best.wavelet}, mode={best.threshold_mode}, "
            f"rule={best.threshold_rule}  |  E[RMSE]={best.rmse_mean:.6f}, "
            f"Var[RMSE]={best.rmse_var:.2e}  [{tag}]  |  Feasibility Rate: {result.feasibility_rate*100:.1f}%"
        )

    return result


# ===================================================================
# Original preprocessing functions (preserved, type-annotated)
# ===================================================================


def bandpass_filter(
    signal_data: Signal, fs: float, lowcut: float, highcut: float, order: int = 5
) -> Signal:
    """
    Aplica un filtro pasabanda Butterworth a la señal.

    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    fs : float
        Frecuencia de muestreo en Hz
    lowcut : float
        Frecuencia de corte inferior en Hz
    highcut : float
        Frecuencia de corte superior en Hz
    order : int, opcional
        Orden del filtro (por defecto 5)

    Retorna:
    --------
    filtered_signal : ndarray
        Señal filtrada
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Asegurar que las frecuencias estén en el rango válido
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))

    if low >= high:
        raise ValueError(
            "La frecuencia de corte inferior debe ser menor que la superior"
        )

    b, a = signal.butter(order, [low, high], btype="band")
    filtered_signal: Signal = signal.filtfilt(b, a, signal_data)

    return filtered_signal


def normalize_signal(signal_data: Signal, method: str = "zscore") -> Signal:
    """
    Normaliza la señal usando diferentes métodos.

    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    method : str, opcional
        Método de normalización: 'zscore', 'minmax', 'robust' (por defecto 'zscore')

    Retorna:
    --------
    normalized_signal : ndarray
        Señal normalizada
    """
    signal_data = np.asarray(signal_data)

    if method == "zscore":
        # Normalización Z-score (media 0, desviación estándar 1)
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std == 0:
            return signal_data - mean
        normalized_signal = (signal_data - mean) / std

    elif method == "minmax":
        # Normalización Min-Max [0, 1]
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        if max_val == min_val:
            return signal_data - min_val
        normalized_signal = (signal_data - min_val) / (max_val - min_val)

    elif method == "robust":
        # Normalización robusta usando mediana y MAD
        median = np.median(signal_data)
        mad = np.median(np.abs(signal_data - median))
        if mad == 0:
            return signal_data - median
        normalized_signal = (signal_data - median) / (1.4826 * mad)

    else:
        raise ValueError(
            f"Método '{method}' no reconocido. Use 'zscore', 'minmax' o 'robust'"
        )

    return normalized_signal


def get_envelope(signal_data: Signal) -> Signal:
    """
    Extrae la envolvente de la señal usando la transformada de Hilbert.

    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada

    Retorna:
    --------
    envelope : ndarray
        Envolvente de la señal
    """
    analytic_signal = hilbert(signal_data)
    envelope = np.abs(analytic_signal)

    return envelope


def wavelet_denoise(
    signal_data: Signal,
    wavelet: str = "db4",
    level: Optional[int] = None,
    threshold_method: str = "hard",
) -> Signal:
    """
    Elimina ruido de la señal usando transformada wavelet.

    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    wavelet : str, opcional
        Tipo de wavelet a usar (por defecto 'db4')
    level : int, opcional
        Nivel de descomposición. Si es None, se calcula automáticamente
    threshold_method : str, opcional
        Método de umbralización: 'soft' o 'hard' (por defecto 'hard')

    Retorna:
    --------
    denoised_signal : ndarray
        Señal con ruido reducido
    """
    signal_data = np.asarray(signal_data)

    # Calcular nivel óptimo si no se proporciona
    if level is None:
        level = min(pywt.dwt_max_level(len(signal_data), wavelet), 6)

    # Descomposición wavelet
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)

    # Calcular umbral usando regla Minimax
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    n = len(signal_data)
    threshold = 0.0 if n <= 32 else sigma * (0.3936 + 0.1829 * np.log2(n))

    # Aplicar umbralización a los coeficientes de detalle
    coeffs_thresholded = [coeffs[0]]  # Mantener coeficientes de aproximación
    for i in range(1, len(coeffs)):
        if threshold_method == "soft":
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode="soft"))
        else:
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode="hard"))

    # Reconstruir señal
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)

    # Ajustar longitud si es necesario
    if len(denoised_signal) > len(signal_data):
        denoised_signal = denoised_signal[: len(signal_data)]
    elif len(denoised_signal) < len(signal_data):
        denoised_signal = np.pad(
            denoised_signal, (0, len(signal_data) - len(denoised_signal)), "edge"
        )

    return denoised_signal


def adaptive_filter_lms(
    signal_data: Signal,
    reference: Optional[Signal] = None,
    mu: float = 0.01,
    filter_order: int = 32,
) -> Signal:
    """
    Filtro adaptativo LMS simple para eliminación de ruido.

    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada contaminada
    reference : array-like, opcional
        Señal de referencia de ruido. Si es None, se usa versión retardada de la señal
    mu : float, opcional
        Tasa de aprendizaje (por defecto 0.01)
    filter_order : int, opcional
        Orden del filtro adaptativo (por defecto 32)

    Retorna:
    --------
    filtered_signal : ndarray
        Señal filtrada
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)

    # Si no hay señal de referencia, usar versión retardada
    if reference is None:
        reference = np.roll(signal_data, filter_order)

    # Inicializar pesos del filtro
    weights = np.zeros(filter_order)
    filtered_signal = np.zeros(n)

    # Algoritmo LMS
    for i in range(filter_order, n):
        # Vector de entrada
        x = reference[i - filter_order : i][::-1]

        # Salida del filtro
        y = np.dot(weights, x)

        # Error
        e = signal_data[i] - y

        # Actualizar pesos
        weights = weights + 2 * mu * e * x

        filtered_signal[i] = e

    return filtered_signal


def preprocess_signal(
    signal_data: Signal,
    fs: float,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    normalize: bool = True,
    envelope: bool = True,
    denoise: bool = True,
) -> Tuple[Signal, Dict[str, Any]]:
    """
    Pipeline completo de preprocesamiento de señal.

    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada en bruto
    fs : float
        Frecuencia de muestreo en Hz
    lowcut : float, opcional
        Frecuencia de corte inferior del filtro pasabanda
    highcut : float, opcional
        Frecuencia de corte superior del filtro pasabanda
    normalize : bool, opcional
        Si aplicar normalización (por defecto True)
    envelope : bool, opcional
        Si extraer la envolvente (por defecto True)
    denoise : bool, opcional
        Si aplicar eliminación de ruido (por defecto True)

    Retorna:
    --------
    processed_signal : ndarray
        Señal procesada
    processing_info : dict
        Información sobre los pasos de procesamiento aplicados
    """
    signal_data = np.asarray(signal_data)
    processing_info = {}

    # Filtro pasabanda si se especifican frecuencias
    if lowcut is not None and highcut is not None:
        signal_data = bandpass_filter(signal_data, fs, lowcut, highcut)
        processing_info["bandpass_filter"] = {"lowcut": lowcut, "highcut": highcut}

    # Normalización
    if normalize:
        signal_data = normalize_signal(signal_data, method="zscore")
        processing_info["normalized"] = True

    # Extracción de envolvente
    if envelope:
        signal_data = get_envelope(signal_data)
        processing_info["envelope"] = True

    # Eliminación de ruido
    if denoise:
        signal_data = wavelet_denoise(signal_data)
        processing_info["denoised"] = True

    return signal_data, processing_info
