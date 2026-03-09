"""
Módulo de validación del algoritmo de detección.

Phase 4 — Quantification:
    1. **Asymptotic time-complexity (Big-O)** of each Phase-3 algorithm
       (Kalman, adaptive EWMA, CUSUM) is measured empirically by timing
       execution across geometrically spaced input sizes and fitting the
       exponent of a power-law model t(n) = a┬╖n^b.
    2. **Confusion matrix** contrasting *convergence latency* (number of
       samples required for the Kalman gain / EWMA α to stabilise within a
       tolerance) against the **false-positive rate** of each tracker,
       parameterised by stochastic variation of the event rate.
    3. Full PEP 484 type annotations throughout.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Signal = NDArray[np.floating[Any]]
Labels = Union[NDArray[np.str_], NDArray[np.object_], List[str]]


# ===================================================================
# Phase-4 — §1  Asymptotic time-complexity measurement (Big-O)
# ===================================================================


@dataclass
class BigOEstimate:
    """Result of empirical Big-O estimation for a single algorithm."""

    algorithm_name: str
    exponent_b: float  # fitted exponent in t(n) = a * n^b
    coefficient_a: float  # fitted coefficient
    r_squared: float  # goodness of fit
    sizes: NDArray[np.int64]  # tested input sizes
    wall_times: NDArray[np.float64]  # measured wall times (seconds)
    big_o_label: str  # human-readable, e.g. "O(n^1.01)"
    exponent_b_ci_low: float = 0.0  # 95% CI lower bound
    exponent_b_ci_high: float = 0.0  # 95% CI upper bound


def _power_law(
    n: NDArray[np.floating[Any]], a: float, b: float
) -> NDArray[np.floating[Any]]:
    """Model function ``t(n) = a * n^b``."""
    return a * np.power(n.astype(np.float64), b)


def measure_algorithm_complexity(
    algorithm_fn: Callable[[Signal], Any],
    sizes: Sequence[int] = (256, 512, 1024, 2048, 4096, 8192, 16384),
    n_repeats: int = 5,
    seed: Optional[int] = 42,
    algorithm_name: str = "unknown",
) -> BigOEstimate:
    """Empirically estimate the Big-O complexity of *algorithm_fn*.

    For each size in *sizes* the function is called *n_repeats* times on
    random Δt vectors and the **median** wall-clock time is recorded.
    A power-law ``t(n) = a┬╖n^b`` is then fitted in log-log space via
    non-linear least squares.

    Parameters
    ----------
    algorithm_fn : callable
        ``f(delta_t: Signal) -> Any``.  The return value is discarded.
    sizes : sequence of int
        Input sizes to benchmark.
    n_repeats : int
        Repetitions per size (takes median).
    seed : int, optional
        RNG seed.
    algorithm_name : str
        Label for the result.

    Returns
    -------
    BigOEstimate
    """
    rng = np.random.default_rng(seed)
    median_times: List[float] = []
    mean_times: List[float] = []
    std_times: List[float] = []
    actual_sizes: List[int] = []

    for n in sizes:
        dt_warmup: Signal = rng.exponential(scale=1e-4, size=n).astype(np.float64)
        # Warmup: run once to populate caches, JIT paths, etc.
        algorithm_fn(dt_warmup)

        times: List[float] = []
        for _ in range(n_repeats):
            dt: Signal = rng.exponential(scale=1e-4, size=n).astype(np.float64)
            t0 = time.perf_counter_ns()
            algorithm_fn(dt)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) * 1e-9)  # convert ns to seconds
        median_times.append(float(np.median(times)))
        mean_times.append(float(np.mean(times)))
        std_times.append(float(np.std(times)))
        actual_sizes.append(n)

    s_arr = np.array(actual_sizes, dtype=np.float64)
    t_arr = np.array(median_times, dtype=np.float64)

    # Fit power law in log-log space
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, _ = curve_fit(
                _power_law,
                s_arr,
                t_arr,
                p0=[1e-8, 1.0],
                maxfev=10000,
            )
            a_fit, b_fit = float(popt[0]), float(popt[1])
        except RuntimeError:
            # Fallback: linear regression in log-log
            log_s = np.log(s_arr)
            log_t = np.log(t_arr + 1e-30)
            slope, intercept, _, _, _ = stats.linregress(log_s, log_t)
            b_fit = float(slope)
            a_fit = float(np.exp(intercept))

    # R² in original space
    t_pred = _power_law(s_arr, a_fit, b_fit)
    ss_res = float(np.sum((t_arr - t_pred) ** 2))
    ss_tot = float(np.sum((t_arr - np.mean(t_arr)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot

    # Bootstrap 95% CI on exponent b
    n_bootstrap = 1000
    boot_exponents: List[float] = []
    for _ in range(n_bootstrap):
        idx_boot = rng.choice(len(s_arr), size=len(s_arr), replace=True)
        s_boot = s_arr[idx_boot]
        t_boot = t_arr[idx_boot]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt_b, _ = curve_fit(
                    _power_law, s_boot, t_boot, p0=[1e-8, 1.0], maxfev=5000
                )
                boot_exponents.append(float(popt_b[1]))
            except RuntimeError:
                log_sb = np.log(s_boot)
                log_tb = np.log(t_boot + 1e-30)
                sl_b, _, _, _, _ = stats.linregress(log_sb, log_tb)
                boot_exponents.append(float(sl_b))

    ci_low = float(np.percentile(boot_exponents, 2.5))
    ci_high = float(np.percentile(boot_exponents, 97.5))
    label = f"O(n^{b_fit:.2f} [CI95: {ci_low:.2f}-{ci_high:.2f}])"

    return BigOEstimate(
        algorithm_name=algorithm_name,
        exponent_b=b_fit,
        coefficient_a=a_fit,
        r_squared=r2,
        sizes=np.array(actual_sizes, dtype=np.int64),
        wall_times=t_arr,
        big_o_label=label,
        exponent_b_ci_low=ci_low,
        exponent_b_ci_high=ci_high,
    )


def measure_all_tracking_complexities(
    sizes: Sequence[int] = (256, 512, 1024, 2048, 4096, 8192, 16384),
    n_repeats: int = 5,
    seed: Optional[int] = 42,
) -> Dict[str, BigOEstimate]:
    """Measure Big-O for every Phase-3 algorithm.

    Returns
    -------
    dict
        ``{'Kalman': BigOEstimate, 'AdaptiveEWMA': …, 'CUSUM': …}``
    """
    # Import here to avoid circular dependency at module level
    from deltapd.trackers import (
        AdaptiveEWMATracker,
        CUSUMDetector,
        KalmanDeltaTTracker,
    )

    kalman = KalmanDeltaTTracker()
    ewma = AdaptiveEWMATracker()
    cusum = CUSUMDetector()

    results: Dict[str, BigOEstimate] = {}

    results["Kalman"] = measure_algorithm_complexity(
        lambda dt: kalman.track(dt),
        sizes=sizes,
        n_repeats=n_repeats,
        seed=seed,
        algorithm_name="Kalman",
    )
    results["AdaptiveEWMA"] = measure_algorithm_complexity(
        lambda dt: ewma.track(dt),
        sizes=sizes,
        n_repeats=n_repeats,
        seed=seed,
        algorithm_name="AdaptiveEWMA",
    )
    results["CUSUM"] = measure_algorithm_complexity(
        lambda dt: cusum.detect(dt),
        sizes=sizes,
        n_repeats=n_repeats,
        seed=seed,
        algorithm_name="CUSUM",
    )

    return results


# ===================================================================
# Phase-4 — §2  Convergence latency & FPR confusion matrix
# ===================================================================


@dataclass
class ConvergenceMetrics:
    """Per-algorithm convergence classification metrics."""

    algorithm_name: str
    convergence_latency: int
    false_positive_rate: float
    precision_score: float
    recall_score: float
    f1_score: float
    total_samples: int


@dataclass
class ConvergenceConfusionMatrix:
    """Confusion-matrix-style comparison across algorithms and event-rate variation levels."""

    algorithms: List[str]
    variation_levels: NDArray[np.float64]
    latency_matrix: NDArray[np.float64]
    fpr_matrix: NDArray[np.float64]
    precision_matrix: NDArray[np.float64]
    recall_matrix: NDArray[np.float64]
    f1_matrix: NDArray[np.float64]
    raw_metrics: List[List[ConvergenceMetrics]]


def _eval_classification(
    delta_t_stat: Signal,
    delta_t_anom: Signal,
    tracker_name: str,
    tracker_cls: Any,
    **kwargs,
) -> Tuple[int, int, int, int]:
    """Evaluate binary classification using a step-change anomaly.

    A shift in event rate is introduced at N/2.
    Alarms before N/2 are False Positives.
    Alarms after N/2 are True Positives.
    """
    n_samples = len(delta_t_anom)
    change_idx = n_samples // 2

    tracker = tracker_cls(**kwargs)
    if tracker_name == "CUSUM":
        res = tracker.detect(delta_t_anom)
        alarms = np.zeros(n_samples, dtype=bool)
        if len(res.alarm_indices) > 0:
            alarms[np.array(res.alarm_indices, dtype=int)] = True
    else:
        res = tracker.track(delta_t_anom)
        residuals = getattr(res, "residuals")
        # Estimate from stationary period
        mu = float(np.mean(residuals[:change_idx]))
        sigma = float(np.std(residuals[:change_idx])) + 1e-30
        alarms = np.abs(residuals - mu) > 3.0 * sigma

    fp = int(np.sum(alarms[:change_idx]))
    tp = int(np.sum(alarms[change_idx:]))
    tn = change_idx - fp
    fn = (n_samples - change_idx) - tp
    return tp, fp, tn, fn


def generate_convergence_confusion_matrix(
    base_rate: float = 1e-4,
    n_samples: int = 2000,
    variation_levels: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0),
    n_monte_carlo: int = 50,
    seed: Optional[int] = 42,
    cusum_threshold: float = 5.0,
    cusum_drift: float = 0.5,
) -> ConvergenceConfusionMatrix:
    """Build a confusion matrix of convergence latency and classification metrics."""
    from deltapd.trackers import AdaptiveEWMATracker, CUSUMDetector, KalmanDeltaTTracker

    rng = np.random.default_rng(seed)
    algo_names = ["Kalman", "AdaptiveEWMA", "CUSUM"]
    n_algos = len(algo_names)
    n_var = len(variation_levels)

    lat_m = np.zeros((n_algos, n_var), dtype=np.float64)
    fpr_m = np.zeros((n_algos, n_var), dtype=np.float64)
    prec_m = np.zeros((n_algos, n_var), dtype=np.float64)
    rec_m = np.zeros((n_algos, n_var), dtype=np.float64)
    f1_m = np.zeros((n_algos, n_var), dtype=np.float64)
    raw_all: List[List[ConvergenceMetrics]] = [[] for _ in range(n_var)]

    for vi, cv in enumerate(variation_levels):
        for _ in range(n_monte_carlo):
            if cv <= 0.0:
                dt_stat = np.full(n_samples, base_rate, dtype=np.float64)
                dt_anom = np.concatenate(
                    [
                        np.full(n_samples // 2, base_rate, dtype=np.float64),
                        np.full(
                            n_samples - n_samples // 2,
                            base_rate * 0.5,
                            dtype=np.float64,
                        ),
                    ]
                )
            else:
                shape = 1.0 / (cv**2)
                dt_stat = rng.gamma(shape, base_rate / shape, size=n_samples).astype(
                    np.float64
                )
                dt_anom = np.concatenate(
                    [
                        rng.gamma(shape, base_rate / shape, size=n_samples // 2),
                        rng.gamma(
                            shape,
                            (base_rate * 0.5) / shape,
                            size=n_samples - n_samples // 2,
                        ),
                    ]
                ).astype(np.float64)

            for ai, (name, cls, kwargs) in enumerate(
                [
                    ("Kalman", KalmanDeltaTTracker, {}),
                    ("AdaptiveEWMA", AdaptiveEWMATracker, {}),
                    (
                        "CUSUM",
                        CUSUMDetector,
                        {"threshold": cusum_threshold, "drift": cusum_drift},
                    ),
                ]
            ):
                tp, fp, tn, fn = _eval_classification(
                    dt_stat, dt_anom, name, cls, **kwargs
                )

                # BUG #1 FIX: Dynamic Latency Proxy
                if name == "CUSUM":
                    tracker = cls(**kwargs)
                    res = tracker.detect(dt_anom)
                    alarm_idx = (
                        res.alarm_indices if hasattr(res, "alarm_indices") else []
                    )
                else:
                    tracker = cls(**kwargs)
                    res = tracker.track(dt_anom)
                    residuals = getattr(res, "residuals")
                    mu = float(np.mean(residuals[: n_samples // 2]))
                    sigma = float(np.std(residuals[: n_samples // 2])) + 1e-30
                    alarms = np.abs(residuals - mu) > 3.0 * sigma
                    alarm_idx = np.where(alarms)[0]

                post_change = np.array(
                    [idx for idx in alarm_idx if idx >= (n_samples // 2)]
                )
                lat = (
                    int(post_change[0] - (n_samples // 2))
                    if len(post_change) > 0
                    else (n_samples - (n_samples // 2))
                )

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                lat_m[ai, vi] += lat
                fpr_m[ai, vi] += fpr
                prec_m[ai, vi] += prec
                rec_m[ai, vi] += rec
                f1_m[ai, vi] += f1

    lat_m /= n_monte_carlo
    fpr_m /= n_monte_carlo
    prec_m /= n_monte_carlo
    rec_m /= n_monte_carlo
    f1_m /= n_monte_carlo

    return ConvergenceConfusionMatrix(
        algorithms=algo_names,
        variation_levels=np.array(variation_levels, dtype=np.float64),
        latency_matrix=lat_m,
        fpr_matrix=fpr_m,
        precision_matrix=prec_m,
        recall_matrix=rec_m,
        f1_matrix=f1_m,
        raw_metrics=raw_all,
    )


def generate_phase4_report(
    complexity: Dict[str, BigOEstimate],
    confusion: ConvergenceConfusionMatrix,
) -> str:
    """Human-readable report including Big-O and classification metrics."""
    lines: List[str] = []
    lines.append("=" * 105)
    lines.append("  PHASE 4 — QUANTIFICATION REPORT")
    lines.append("=" * 105)
    lines.append("")
    lines.append("§1  ASYMPTOTIC TIME-COMPLEXITY (empirical Big-O)")
    lines.append("-" * 105)
    for name, est in complexity.items():
        lines.append(
            f"  {name:<18s}  {est.big_o_label:<40s}  a={est.coefficient_a:.2e}  R\u00b2={est.r_squared:.4f}"
        )

    lines.append("")
    lines.append("\u00a72  CLASSIFICATION METRICS (Precision, Recall, F1) & FPR MATRIX")
    lines.append("-" * 105)

    hdr = f"  {'CV(dt)':<10s}"
    for alg in confusion.algorithms:
        hdr += f" | {alg:>24s}"
    lines.append(hdr)

    subhdr = f"  {'':<10s}"
    for _ in confusion.algorithms:
        subhdr += f" | {'FPR':>5s} {'Prec':>5s} {'Rec':>5s} {'F1':>5s}"
    lines.append(subhdr)
    lines.append("  " + "-" * (10 + len(confusion.algorithms) * 27))

    for vi, cv in enumerate(confusion.variation_levels):
        row = f"  {cv:<10.2f}"
        for ai in range(len(confusion.algorithms)):
            fpr = confusion.fpr_matrix[ai, vi]
            prec = confusion.precision_matrix[ai, vi]
            rec = confusion.recall_matrix[ai, vi]
            f1 = confusion.f1_matrix[ai, vi]
            row += f" | {fpr:5.3f} {prec:5.3f} {rec:5.3f} {f1:5.3f}"
        lines.append(row)

    # §3 — Operational Metrics
    lines.append("")
    lines.append("\u00a73  OPERATIONAL METRICS")
    lines.append("-" * 105)
    lines.append(
        f"  {'Algorithm':<18s}  {'Avg Latency (samples)':<24s}  {'Avg FPR':<12s}  {'FAR/min (est)':<14s}"
    )
    lines.append("  " + "-" * 70)
    for ai, alg in enumerate(confusion.algorithms):
        avg_lat = float(np.mean(confusion.latency_matrix[ai, :]))
        avg_fpr = float(np.mean(confusion.fpr_matrix[ai, :]))
        # FAR per minute: assume 1 second observation window
        far_per_min = avg_fpr * 60.0
        lines.append(
            f"  {alg:<18s}  {avg_lat:<24.1f}  {avg_fpr:<12.4f}  {far_per_min:<14.2f}"
        )

    lines.append("")
    lines.append("=" * 105)
    return "\n".join(lines)
