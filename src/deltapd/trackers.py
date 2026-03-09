"""
Módulo de algoritmos ciegos para comparación y seguimiento de Δt.

Phase 3 — Tracking Evaluation:
    Three exclusive algorithms are applied to the Δt vector produced by
    Phase 2 (descriptors.py):

    1. **1-D Kalman Filter** — linear state-space tracker for the inter-pulse
       interval, with online estimation of process / measurement noise.
    2. **Adaptive EWMA** — exponentially weighted moving average with a
       time-varying smoothing factor that adapts to the local variance of Δt.
    3. **CUSUM** — cumulative sum change-point detector that flags shifts in
       the mean event rate.

    The entry point :func:`apply_delta_t_tracking` runs all three algorithms
    on a given Δt vector and returns a unified result dictionary.

Legacy filters (EWMA, SimpleMovingAverage, KalmanFilter1D, AdaptiveLMS,
AdaptiveRLS) are preserved at module level below for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Signal = NDArray[np.floating[Any]]


# ===================================================================
# Phase 3 — Δt Tracking Algorithms  (PRIMARY INTERFACE)
# ===================================================================

# -------------------------------------------------------------------
# 3.1  1-D Kalman Filter for Δt tracking
# -------------------------------------------------------------------


@dataclass
class KalmanDeltaTResult:
    """Result container for the Kalman Δt tracker."""

    filtered: Signal  # Kalman-smoothed Δt estimates
    residuals: Signal  # Innovation (measurement ΓêÆ prediction)
    kalman_gains: Signal  # Gain sequence
    z_scores: Signal  # Normalized innovation (Z-score)
    steady_state_gain: float  # Final Kalman gain


class KalmanDeltaTTracker:
    """1-D Kalman filter specialised for inter-pulse interval tracking.

    State model::

        x[k] = x[k-1] + w[k],   w ~ N(0, Q)
        z[k] = x[k]   + v[k],   v ~ N(0, R)

    where *x* is the latent (true) Δt and *z* is the observed noisy Δt.

    Parameters
    ----------
    process_variance : float
        Process noise variance *Q*.
    measurement_variance : float
        Measurement noise variance *R*.
    initial_estimate : float or None
        If ``None``, the first measurement is used.
    initial_error : float
        Initial estimation-error covariance *PΓéÇ*.
    """

    def __init__(
        self,
        process_variance: float = 7.4e-14,  # Q escalado temporalmente por alfa=741.5 (empírico NIS)
        measurement_variance: float = 1.85e-15,  # R escalado por alfa=741.5 (cuantización + Jitter global)
        initial_estimate: Optional[float] = None,
        initial_error: float = 1e-12,     # P0: Incertidumbre inicial (1 us de std_dev)
    ) -> None:
        self.Q: float = process_variance
        self.R: float = measurement_variance
            
        self._x0: Optional[float] = initial_estimate
        self._P0: float = initial_error

    # ---- public API ------------------------------------------------

    def track(self, delta_t: Signal) -> KalmanDeltaTResult:
        """Run the Kalman filter over the full Δt vector.

        Parameters
        ----------
        delta_t : Signal
            1-D vector of observed inter-pulse intervals.

        Returns
        -------
        KalmanDeltaTResult
        """
        n: int = len(delta_t)
        x_hat = np.empty(n, dtype=np.float64)
        residuals = np.empty(n, dtype=np.float64)
        gains = np.empty(n, dtype=np.float64)
        z_scores = np.empty(n, dtype=np.float64)

        # Initialise
        x: float = delta_t[0] if self._x0 is None else self._x0
        P: float = self._P0

        for k in range(n):
            dt_val = float(delta_t[k])
            
            # [RE-INYECCIÓN DE COVARIANZA]
            # Si el intervalo supera los 10 segundos, asumimos un gap macro-temporal 
            # (osciloscopio ciego) y reseteamos la matriz de incertidumbre.
            if dt_val > 10.0:
                P = self._P0
                # También es físicamente pragmático resetear el estado 
                # a la observación actual tras un apagón largo.
                x = dt_val 

            # --- Predict ---
            x_pred: float = x
            P_pred: float = P + self.Q

            # --- Update ---
            innovation: float = dt_val - x_pred
            S: float = P_pred + self.R
            K: float = P_pred / S

            x = x_pred + K * innovation
            P = (1.0 - K) * P_pred

            x_hat[k] = x
            residuals[k] = innovation
            gains[k] = K
            
            # [EXTRACCIÓN ESTOCÁSTICA] Z-Score de la innovación
            z_scores[k] = innovation / np.sqrt(S) if S > 0 else 0.0

        return KalmanDeltaTResult(
            filtered=x_hat,
            residuals=residuals,
            kalman_gains=gains,
            z_scores=z_scores,
            steady_state_gain=float(gains[-1]),
        )


# -------------------------------------------------------------------
# 3.2  Adaptive EWMA for Δt tracking
# -------------------------------------------------------------------


@dataclass
class AdaptiveEWMAResult:
    """Result container for the adaptive EWMA tracker."""

    smoothed: Signal  # EWMA-smoothed Δt
    residuals: Signal  # observation ΓêÆ smoothed
    alpha_sequence: Signal  # Time-varying smoothing factor


class AdaptiveEWMATracker:
    """EWMA with an adaptive smoothing factor driven by local variance.

    The smoothing factor α[k] is adjusted at every step:

    .. math::

        \\alpha[k] = \\text{clip}\\!\\left(
            \\alpha_0 \\cdot \\frac{\\sigma_{\\text{local}}^2[k]}
                                     {\\sigma_{\\text{global}}^2},
            \\; \\alpha_{\\min}, \\; \\alpha_{\\max}
        \\right)

    This puts more weight on recent observations when local variability
    increases (i.e. the event rate is shifting).

    Parameters
    ----------
    alpha_0 : float
        Baseline smoothing factor.
    alpha_min : float
        Lower bound for α.
    alpha_max : float
        Upper bound for α.
    variance_window : int
        Window size for local variance estimation.
    """

    def __init__(
        self,
        alpha_0: float = 0.2,
        alpha_min: float = 0.05,
        alpha_max: float = 0.8,
        variance_window: int = 10,
    ) -> None:
        self.alpha_0 = alpha_0
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.var_win = variance_window

    def track(self, delta_t: Signal) -> AdaptiveEWMAResult:
        """Run the adaptive EWMA over the full Δt vector.

        Parameters
        ----------
        delta_t : Signal
            1-D vector of inter-pulse intervals.

        Returns
        -------
        AdaptiveEWMAResult
        """
        n: int = len(delta_t)
        smoothed = np.empty(n, dtype=np.float64)
        alphas = np.empty(n, dtype=np.float64)
        residuals = np.empty(n, dtype=np.float64)

        global_var: float = float(np.var(delta_t)) + 1e-30
        s: float = float(delta_t[0])

        for k in range(n):
            # Local variance (causal window)
            lo = max(0, k - self.var_win + 1)
            local_var: float = float(np.var(delta_t[lo : k + 1])) + 1e-30

            alpha: float = np.clip(
                self.alpha_0 * (local_var / global_var),
                self.alpha_min,
                self.alpha_max,
            )

            s = alpha * float(delta_t[k]) + (1.0 - alpha) * s

            smoothed[k] = s
            alphas[k] = alpha
            residuals[k] = float(delta_t[k]) - s

        return AdaptiveEWMAResult(
            smoothed=smoothed,
            residuals=residuals,
            alpha_sequence=alphas,
        )


# -------------------------------------------------------------------
# 3.3  CUSUM change-point detector for Δt
# -------------------------------------------------------------------


@dataclass
class CUSUMResult:
    """Result container for the CUSUM detector."""

    g_plus: Signal  # Upper cumulative sum
    g_minus: Signal  # Lower cumulative sum
    alarms: NDArray[np.bool_]  # Boolean alarm vector
    alarm_indices: NDArray[np.intp]
    n_alarms: int
    threshold: float
    drift: float


class CUSUMDetector:
    """Two-sided CUSUM (Page 1954) for detecting mean-shifts in Δt.

    The detector maintains two cumulative sums:

    .. math::

        g^+[k] = \\max(0,\\; g^+[k-1] + z[k] - \\mu_0 - \\delta)

        g^-[k] = \\max(0,\\; g^-[k-1] - z[k] + \\mu_0 - \\delta)

    An alarm is raised when either statistic exceeds the threshold *h*.

    Parameters
    ----------
    threshold : float
        Decision threshold *h*.
    drift : float
        Allowable drift *╬┤* (half the minimum detectable shift).
    mu_0 : float or None
        Target (in-control) mean.  If ``None``, estimated from data.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.5,
        mu_0: Optional[float] = None,
    ) -> None:
        self.threshold = threshold
        self.drift = drift
        self._mu_0 = mu_0

    def detect(self, z_scores: Signal, burnin: int = 200) -> CUSUMResult:
        """Run CUSUM over the standardized innovation (Z-scores) vector.

        Parameters
        ----------
        z_scores : Signal
            1-D vector of normalized residuals (Z-scores) from the Kalman filter.
        burnin : int
            Number of initial samples used to estimate the in-control mean (mu_0).
            The empirical mean of z_scores[:burnin] is subtracted before accumulation
            to remove Kalman startup bias.

        Returns
        -------
        CUSUMResult
        """
        n: int = len(z_scores)

        # Estimate in-control mean from the burn-in window
        n0 = min(burnin, n)
        mu_0 = float(np.mean(z_scores[:n0])) if self._mu_0 is None else self._mu_0

        g_plus = np.zeros(n, dtype=np.float64)
        g_minus = np.zeros(n, dtype=np.float64)
        alarms = np.zeros(n, dtype=np.bool_)

        for k in range(1, n):
            z = float(z_scores[k]) - mu_0
            g_plus[k] = max(0.0, g_plus[k - 1] + z - self.drift)
            g_minus[k] = max(0.0, g_minus[k - 1] - z - self.drift)
            if g_plus[k] > self.threshold or g_minus[k] > self.threshold:
                alarms[k] = True

        alarm_idx = np.nonzero(alarms)[0]

        return CUSUMResult(
            g_plus=g_plus,
            g_minus=g_minus,
            alarms=alarms,
            alarm_indices=alarm_idx,
            n_alarms=int(alarm_idx.size),
            threshold=self.threshold,
            drift=self.drift,
        )


# -------------------------------------------------------------------
# 3.*  Unified entry point
# -------------------------------------------------------------------


@dataclass
class DeltaTTrackingResult:
    """Aggregated result of all three Δt-tracking algorithms."""

    kalman: KalmanDeltaTResult
    ewma: AdaptiveEWMAResult
    cusum: CUSUMResult


def apply_delta_t_tracking(
    delta_t: Signal,
    fs: float = 5e9,
    kalman_Q: float = 7.4e-14,
    kalman_R: float = 1.85e-15,
    ewma_alpha: float = 0.2,
    ewma_var_window: int = 10,
    cusum_threshold: float = 8.0,
    cusum_drift: float = 0.5,
) -> DeltaTTrackingResult:
    """Apply all three Phase-3 tracking algorithms to a Δt vector.

    Parameters
    ----------
    delta_t : Signal
        1-D inter-pulse interval vector (output of Phase 2).
    fs : float
        Sampling frequency in Hz (used for Kalman quantization noise).
    kalman_Q : float
        Kalman process noise variance.
    kalman_R : float
        Kalman measurement noise variance.
    ewma_alpha : float
        Baseline EWMA smoothing factor.
    ewma_var_window : int
        Window for local variance in adaptive EWMA.
    cusum_threshold, cusum_drift : float
        CUSUM parameters (applied to standardized Z-scores).

    Returns
    -------
    DeltaTTrackingResult
    """
    kalman_tracker = KalmanDeltaTTracker(
        process_variance=kalman_Q,
        measurement_variance=kalman_R,
    )
    kalman = kalman_tracker.track(delta_t)

    ewma = AdaptiveEWMATracker(
        alpha_0=ewma_alpha,
        variance_window=ewma_var_window,
    ).track(delta_t)

    # Feed the scale-invariant Z-scores from the Kalman filter to CUSUM
    # instead of the raw delta_t vector. This ensures the theoretical drift 
    # of 0.5 operates safely on Gaussian standard deviations.
    cusum = CUSUMDetector(
        threshold=cusum_threshold,
        drift=cusum_drift,
    ).detect(kalman.z_scores)

    return DeltaTTrackingResult(kalman=kalman, ewma=ewma, cusum=cusum)


# ===================================================================
# Legacy algorithms (preserved, type-annotated)
# ===================================================================


class EWMA:
    """
    Exponentially Weighted Moving Average (EWMA).
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.ewma_value = None

    def update(self, value: float) -> float:
        if self.ewma_value is None:
            self.ewma_value = value
        else:
            self.ewma_value = self.alpha * value + (1 - self.alpha) * self.ewma_value

        return self.ewma_value

    def process_signal(self, signal_data: Signal) -> Signal:
        """Process a complete signal through EWMA."""
        self.ewma_value = None
        ewma_signal = np.zeros(len(signal_data))

        for i, value in enumerate(signal_data):
            ewma_signal[i] = self.update(value)

        return ewma_signal

    def calculate_score(self, signal_data: Signal) -> float:
        """Anomaly score based on RMS of EWMA residuals."""
        ewma_signal = self.process_signal(signal_data)
        residuals = signal_data - ewma_signal

        # Usar RMS del residuo como puntaje
        score = np.sqrt(np.mean(residuals**2))

        return score


class SimpleMovingAverage:
    """
    Media móvil simple (SMA).
    """

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self.buffer = []

    def update(self, value):
        """
        Actualiza la media móvil con un nuevo valor.

        Parámetros:
        -----------
        value : float
            Nuevo valor

        Retorna:
        --------
        sma : float
            Media móvil actualizada
        """
        self.buffer.append(value)

        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        return np.mean(self.buffer)

    def process_signal(self, signal_data):
        """
        Procesa una señal completa.

        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada

        Retorna:
        --------
        sma_signal : ndarray
            Señal filtrada con media móvil
        """
        self.buffer = []
        sma_signal = np.zeros(len(signal_data))

        for i, value in enumerate(signal_data):
            sma_signal[i] = self.update(value)

        return sma_signal

    def calculate_score(self, signal_data):
        """
        Calcula un puntaje de anomalía.

        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada

        Retorna:
        --------
        score : float
            Puntaje de anomalía
        """
        sma_signal = self.process_signal(signal_data)
        residuals = signal_data - sma_signal

        score = np.sqrt(np.mean(residuals**2))

        return score


class KalmanFilter1D:
    """
    Filtro de Kalman 1D simple.
    """

    def __init__(
        self,
        process_variance=1e-5,
        measurement_variance=1e-2,
        initial_estimate=0.0,
        initial_error=1.0,
    ):
        """
        Inicializa el filtro de Kalman.

        Parámetros:
        -----------
        process_variance : float, opcional
            Varianza del ruido del proceso (Q)
        measurement_variance : float, opcional
            Varianza del ruido de medición (R)
        initial_estimate : float, opcional
            Estimación inicial del estado
        initial_error : float, opcional
            Error de estimación inicial
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error = initial_error

    def update(self, measurement):
        """
        Actualiza el filtro con una nueva medición.

        Parámetros:
        -----------
        measurement : float
            Nueva medición

        Retorna:
        --------
        estimate : float
            Estimación filtrada
        """
        # Predicción
        prediction_error = self.error + self.process_variance

        # Actualización
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error = (1 - kalman_gain) * prediction_error

        return self.estimate

    def process_signal(self, signal_data):
        """
        Procesa una señal completa.

        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada

        Retorna:
        --------
        filtered_signal : ndarray
            Señal filtrada con Kalman
        """
        # Reiniciar con el primer valor
        self.estimate = signal_data[0]
        self.error = 1.0

        filtered_signal = np.zeros(len(signal_data))

        for i, measurement in enumerate(signal_data):
            filtered_signal[i] = self.update(measurement)

        return filtered_signal

    def calculate_score(self, signal_data):
        """
        Calcula un puntaje de anomalía.

        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada

        Retorna:
        --------
        score : float
            Puntaje de anomalía basado en innovación
        """
        filtered_signal = self.process_signal(signal_data)
        innovations = signal_data - filtered_signal

        score = np.sqrt(np.mean(innovations**2))

        return score


class AdaptiveLMS:
    """
    Filtro adaptativo LMS (Least Mean Squares).
    """

    def __init__(self, filter_order=8, mu=0.01):
        """
        Inicializa el filtro LMS.

        Parámetros:
        -----------
        filter_order : int, opcional
            Orden del filtro (por defecto 8)
        mu : float, opcional
            Tasa de aprendizaje (por defecto 0.01)
        """
        self.filter_order = filter_order
        self.mu = mu
        self.weights = np.zeros(filter_order)
        self.buffer = np.zeros(filter_order)

    def update(self, desired, input_sample):
        """
        Actualiza el filtro con nueva muestra.

        Parámetros:
        -----------
        desired : float
            Señal deseada
        input_sample : float
            Muestra de entrada

        Retorna:
        --------
        output : float
            Salida filtrada
        error : float
            Error de predicción
        """
        # Actualizar buffer
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = input_sample

        # Calcular salida
        output = np.dot(self.weights, self.buffer)

        # Calcular error
        error = desired - output

        # Actualizar pesos con límite para evitar divergencia
        self.weights = self.weights + 2 * self.mu * error * self.buffer

        # Limitar magnitud de los pesos para evitar divergencia
        max_weight = 10.0
        self.weights = np.clip(self.weights, -max_weight, max_weight)

        return output, error

    def process_signal(self, signal_data):
        """
        Procesa una señal completa.

        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada

        Retorna:
        --------
        filtered_signal : ndarray
            Señal filtrada
        errors : ndarray
            Errores de predicción
        """
        signal_data = np.asarray(signal_data)
        n = len(signal_data)

        # Reiniciar
        self.weights = np.zeros(self.filter_order)
        self.buffer = np.zeros(self.filter_order)

        filtered_signal = np.zeros(n)
        errors = np.zeros(n)

        # Usar señal retardada como referencia
        for i in range(n):
            if i < self.filter_order:
                filtered_signal[i] = signal_data[i]
                errors[i] = 0
            else:
                output, error = self.update(signal_data[i], signal_data[i - 1])
                filtered_signal[i] = output
                errors[i] = error

        return filtered_signal, errors

    def calculate_score(self, signal_data):
        """
        Calcula un puntaje de anomalía.

        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada

        Retorna:
        --------
        score : float
            Puntaje basado en error de predicción
        """
        _, errors = self.process_signal(signal_data)

        # Usar RMS del error
        score = np.sqrt(np.mean(errors[self.filter_order :] ** 2))

        return score


class AdaptiveRLS:
    """
    Filtro adaptativo RLS (Recursive Least Squares) simplificado.
    """

    def __init__(self, filter_order=8, forgetting_factor=0.99):
        """
        Inicializa el filtro RLS.

        Parámetros:
        -----------
        filter_order : int, opcional
            Orden del filtro (por defecto 8)
        forgetting_factor : float, opcional
            Factor de olvido (lambda), típicamente 0.95-0.99
        """
        self.filter_order = filter_order
        self.lambda_factor = forgetting_factor
        self.weights = np.zeros(filter_order)
        self.P = np.eye(filter_order) * 1000  # Matriz de covarianza inversa
        self.buffer = np.zeros(filter_order)

    def update(self, desired, input_sample):
        """
        Actualiza el filtro con nueva muestra.

        Parámetros:
        -----------
        desired : float
            Señal deseada
        input_sample : float
            Muestra de entrada

        Retorna:
        --------
        output : float
            Salida filtrada
        error : float
            Error de predicción
        """
        # Actualizar buffer
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = input_sample

        # Calcular salida
        output = np.dot(self.weights, self.buffer)

        # Calcular error a priori
        error = desired - output

        # Cálculo del gain vector
        k = (self.P @ self.buffer) / (
            self.lambda_factor + self.buffer @ self.P @ self.buffer
        )

        # Actualizar pesos
        self.weights = self.weights + k * error

        # Actualizar matriz P
        self.P = (self.P - np.outer(k, self.buffer @ self.P)) / self.lambda_factor

        return output, error

    def process_signal(self, signal_data):
        """
        Procesa una señal completa.

        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada

        Retorna:
        --------
        filtered_signal : ndarray
            Señal filtrada
        errors : ndarray
            Errores de predicción
        """
        signal_data = np.asarray(signal_data)
        n = len(signal_data)

        # Reiniciar
        self.weights = np.zeros(self.filter_order)
        self.P = np.eye(self.filter_order) * 1000
        self.buffer = np.zeros(self.filter_order)

        filtered_signal = np.zeros(n)
        errors = np.zeros(n)

        for i in range(n):
            if i < self.filter_order:
                filtered_signal[i] = signal_data[i]
                errors[i] = 0
            else:
                output, error = self.update(signal_data[i], signal_data[i - 1])
                filtered_signal[i] = output
                errors[i] = error

        return filtered_signal, errors

    def calculate_score(self, signal_data):
        """
        Calcula un puntaje de anomalía.

        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada

        Retorna:
        --------
        score : float
            Puntaje basado en error de predicción
        """
        _, errors = self.process_signal(signal_data)

        score = np.sqrt(np.mean(errors[self.filter_order :] ** 2))

        return score


def compare_algorithms(
    signal_data: Signal,
    algorithms: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compara el rendimiento de múltiples algoritmos ciegos.

    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    algorithms : dict, opcional
        Diccionario de algoritmos {nombre: instancia}
        Si es None, usa todos los algoritmos predeterminados

    Retorna:
    --------
    results : dict
        Resultados comparativos por algoritmo
    """
    if algorithms is None:
        algorithms = {
            "EWMA": EWMA(alpha=0.2),
            "SMA": SimpleMovingAverage(window_size=10),
            "Kalman": KalmanFilter1D(),
            "LMS": AdaptiveLMS(filter_order=8, mu=0.01),
            "RLS": AdaptiveRLS(filter_order=8),
        }

    results = {}

    for name, algorithm in algorithms.items():
        score = algorithm.calculate_score(signal_data)
        results[name] = {"score": score, "algorithm": name}

    return results
