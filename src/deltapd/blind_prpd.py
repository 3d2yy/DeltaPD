from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp


@dataclass
class BlindPRPDCandidate:
    method: str
    freq_hz: float
    score: float
    coherence: float
    axial_entropy_score: float
    sharpness: float
    half_height_width_hz: float
    score_prominence: float
    common_axial_peak_freq_hz: float = float("nan")
    common_axial_peak_offset_hz: float = float("nan")
    common_axial_sharpness: float = float("nan")
    common_axial_half_height_width_hz: float = float("nan")
    common_axial_prominence: float = float("nan")
    common_axial_width_ratio: float = float("nan")
    common_axial_confidence: float = float("nan")
    bootstrap_iterations: int = 0
    bootstrap_sample_fraction: float = float("nan")
    bootstrap_freq_mean_hz: float = float("nan")
    bootstrap_freq_std_hz: float = float("nan")
    bootstrap_ci_low_hz: float = float("nan")
    bootstrap_ci_high_hz: float = float("nan")
    bootstrap_ci_width_hz: float = float("nan")
    bootstrap_method_agreement: float = float("nan")
    bootstrap_selected_method_counts: dict[str, int] = field(default_factory=dict)
    local_window_count: int = 0
    local_window_size_events: int = 0
    local_window_step_events: int = 0
    local_freq_mean_hz: float = float("nan")
    local_freq_std_hz: float = float("nan")
    local_freq_min_hz: float = float("nan")
    local_freq_max_hz: float = float("nan")
    local_freq_span_hz: float = float("nan")
    local_method_agreement: float = float("nan")
    local_common_confidence_mean: float = float("nan")
    local_dominant_method: str = ""
    local_selected_method_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlindPRPDCalibrationResult:
    requested_method: str
    selected_method: str
    freq_hz: float
    score: float
    coherence: float
    axial_entropy_score: float
    sharpness: float
    half_height_width_hz: float
    score_prominence: float
    common_axial_peak_freq_hz: float = float("nan")
    common_axial_peak_offset_hz: float = float("nan")
    common_axial_sharpness: float = float("nan")
    common_axial_half_height_width_hz: float = float("nan")
    common_axial_prominence: float = float("nan")
    common_axial_width_ratio: float = float("nan")
    common_axial_confidence: float = float("nan")
    bootstrap_iterations: int = 0
    bootstrap_sample_fraction: float = float("nan")
    bootstrap_freq_mean_hz: float = float("nan")
    bootstrap_freq_std_hz: float = float("nan")
    bootstrap_ci_low_hz: float = float("nan")
    bootstrap_ci_high_hz: float = float("nan")
    bootstrap_ci_width_hz: float = float("nan")
    bootstrap_method_agreement: float = float("nan")
    bootstrap_selected_method_counts: dict[str, int] = field(default_factory=dict)
    local_window_count: int = 0
    local_window_size_events: int = 0
    local_window_step_events: int = 0
    local_freq_mean_hz: float = float("nan")
    local_freq_std_hz: float = float("nan")
    local_freq_min_hz: float = float("nan")
    local_freq_max_hz: float = float("nan")
    local_freq_span_hz: float = float("nan")
    local_method_agreement: float = float("nan")
    local_common_confidence_mean: float = float("nan")
    local_dominant_method: str = ""
    local_selected_method_counts: dict[str, int] = field(default_factory=dict)
    candidate_spread_hz: float = float("nan")
    winner_margin: float = float("nan")
    candidate_methods: tuple[BlindPRPDCandidate, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlindPRPDInfo:
    calibrated_freq_hz: float
    coherence: float
    phase_offset_deg: float
    display_shift_deg: float
    n_events_used: int
    method: str = "coherence"
    requested_method: str = "coherence"
    selected_method: str = "coherence"
    score: float = float("nan")
    axial_entropy_score: float = float("nan")
    sharpness: float = float("nan")
    half_height_width_hz: float = float("nan")
    score_prominence: float = float("nan")
    common_axial_peak_freq_hz: float = float("nan")
    common_axial_peak_offset_hz: float = float("nan")
    common_axial_sharpness: float = float("nan")
    common_axial_half_height_width_hz: float = float("nan")
    common_axial_prominence: float = float("nan")
    common_axial_width_ratio: float = float("nan")
    common_axial_confidence: float = float("nan")
    bootstrap_iterations: int = 0
    bootstrap_sample_fraction: float = float("nan")
    bootstrap_freq_mean_hz: float = float("nan")
    bootstrap_freq_std_hz: float = float("nan")
    bootstrap_ci_low_hz: float = float("nan")
    bootstrap_ci_high_hz: float = float("nan")
    bootstrap_ci_width_hz: float = float("nan")
    bootstrap_method_agreement: float = float("nan")
    bootstrap_selected_method_counts: dict[str, int] = field(default_factory=dict)
    local_window_count: int = 0
    local_window_size_events: int = 0
    local_window_step_events: int = 0
    local_freq_mean_hz: float = float("nan")
    local_freq_std_hz: float = float("nan")
    local_freq_min_hz: float = float("nan")
    local_freq_max_hz: float = float("nan")
    local_freq_span_hz: float = float("nan")
    local_method_agreement: float = float("nan")
    local_common_confidence_mean: float = float("nan")
    local_dominant_method: str = ""
    local_selected_method_counts: dict[str, int] = field(default_factory=dict)
    candidate_spread_hz: float = float("nan")
    winner_margin: float = float("nan")
    candidate_methods: tuple[BlindPRPDCandidate, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prepare_weights(peaks: np.ndarray | None, n_events: int) -> np.ndarray:
    if peaks is None or len(peaks) != n_events:
        return np.ones(n_events, dtype=np.float64)
    weights = np.sqrt(np.abs(np.asarray(peaks, dtype=np.float64)))
    weights[~np.isfinite(weights)] = 0.0
    if np.all(weights <= 0):
        return np.ones(n_events, dtype=np.float64)
    return weights


def _deterministic_subsample(
    toa_s: np.ndarray,
    weights: np.ndarray,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(toa_s) <= max_events:
        return toa_s, weights
    idx = np.linspace(0, len(toa_s) - 1, max_events, dtype=int)
    return toa_s[idx], weights[idx]


def _coherence_value(freq_hz: float, toa_s: np.ndarray, weights: np.ndarray) -> float:
    phasor = np.exp(1j * 4.0 * np.pi * float(freq_hz) * toa_s)
    return float(np.abs(np.sum(weights * phasor)) / max(np.sum(weights), 1e-30))


def _harmonic_power_value(
    freq_hz: float,
    toa_s: np.ndarray,
    weights: np.ndarray,
    *,
    n_harmonics: int,
) -> float:
    base = np.exp(1j * 4.0 * np.pi * float(freq_hz) * toa_s)
    harmonic = base.copy()
    denom = max(np.sum(weights), 1e-30)
    score = 0.0
    for harmonic_idx in range(1, max(int(n_harmonics), 1) + 1):
        resultant = np.abs(np.sum(weights * harmonic)) / denom
        score += (resultant * resultant) / float(harmonic_idx)
        harmonic *= base
    return float(score)


def _epoch_folding_value(
    freq_hz: float,
    toa_s: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int,
    n_shifts: int = 4,
) -> float:
    n_bins = max(int(n_bins), 8)
    phases_unit = np.mod(np.asarray(toa_s, dtype=np.float64) * 2.0 * float(freq_hz), 1.0)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0 or len(phases_unit) < n_bins:
        return 0.0
    expected = total_weight / float(n_bins)
    bin_width = 1.0 / float(n_bins)
    best_score = 0.0
    for shift_idx in range(max(int(n_shifts), 1)):
        shift = shift_idx * bin_width / max(int(n_shifts), 1)
        shifted = np.mod(phases_unit + shift, 1.0)
        hist, _ = np.histogram(
            shifted,
            bins=n_bins,
            range=(0.0, 1.0),
            weights=np.asarray(weights, dtype=np.float64),
        )
        chi2 = float(np.sum((hist - expected) ** 2 / max(expected, 1e-30)))
        if chi2 > best_score:
            best_score = chi2
    return best_score


def _phase_distance_correlation_value(
    freq_hz: float,
    toa_s: np.ndarray,
    values: np.ndarray,
) -> float:
    phases_unit = np.mod(np.asarray(toa_s, dtype=np.float64) * float(freq_hz), 1.0)
    values = np.asarray(values, dtype=np.float64)
    if len(phases_unit) < 12 or len(values) != len(phases_unit):
        return 0.0

    phase_diff = np.abs(phases_unit[:, None] - phases_unit[None, :])
    phase_diff = np.minimum(phase_diff, 1.0 - phase_diff)
    phase_dist = phase_diff * (1.0 - phase_diff)

    value_dist = np.abs(values[:, None] - values[None, :])

    phase_centered = phase_dist - phase_dist.mean(axis=0, keepdims=True) - phase_dist.mean(axis=1, keepdims=True) + phase_dist.mean()
    value_centered = value_dist - value_dist.mean(axis=0, keepdims=True) - value_dist.mean(axis=1, keepdims=True) + value_dist.mean()

    numerator = float(np.sum(phase_centered * value_centered))
    denom_left = float(np.sum(phase_centered * phase_centered))
    denom_right = float(np.sum(value_centered * value_centered))
    denom = np.sqrt(max(denom_left * denom_right, 0.0))
    if denom <= 1e-30:
        return 0.0
    return float(max(numerator / denom, 0.0))


def _gregory_loredo_value(
    freq_hz: float,
    toa_s: np.ndarray,
    *,
    m_min: int = 2,
    m_max: int = 12,
    phase_substeps: int = 4,
) -> float:
    toa = np.asarray(toa_s, dtype=np.float64)
    if len(toa) < 12:
        return float("-inf")

    phases_unit = np.mod(toa * float(freq_hz), 1.0)
    n_events = int(len(phases_unit))
    log_terms_per_m: list[float] = []

    for m_bins in range(max(int(m_min), 2), max(int(m_max), 2) + 1):
        log_occam = -float(gammaln(n_events + m_bins) - gammaln(n_events + 1) - gammaln(m_bins))
        shift_logs: list[float] = []
        for shift_idx in range(max(int(phase_substeps), 1)):
            shift = shift_idx / float(max(int(phase_substeps), 1) * m_bins)
            shifted = np.mod(phases_unit + shift, 1.0)
            counts, _ = np.histogram(shifted, bins=m_bins, range=(0.0, 1.0))
            log_w = float(gammaln(n_events + 1) - np.sum(gammaln(counts + 1)))
            shift_logs.append(n_events * np.log(m_bins) - log_w + log_occam)
        log_terms_per_m.append(float(logsumexp(np.asarray(shift_logs)) - np.log(len(shift_logs))))

    if not log_terms_per_m:
        return float("-inf")
    return float(logsumexp(np.asarray(log_terms_per_m)))


def _phase_distance_correlation_precompute(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    value_dist = np.abs(values[:, None] - values[None, :])
    value_centered = (
        value_dist
        - value_dist.mean(axis=0, keepdims=True)
        - value_dist.mean(axis=1, keepdims=True)
        + value_dist.mean()
    )
    denom_right = float(np.sum(value_centered * value_centered))
    return value_centered, denom_right


def _coherence_curve(
    freq_grid: np.ndarray,
    toa_s: np.ndarray,
    weights: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    scores = np.empty(len(freq_grid), dtype=np.float64)
    denom = max(np.sum(weights), 1e-30)
    toa_row = toa_s[None, :]
    weights_row = weights[None, :]
    for start in range(0, len(freq_grid), chunk_size):
        stop = min(start + chunk_size, len(freq_grid))
        freq_block = freq_grid[start:stop, None]
        phase_block = np.exp(1j * 4.0 * np.pi * freq_block * toa_row)
        scores[start:stop] = np.abs(np.sum(phase_block * weights_row, axis=1)) / denom
    return scores


def _epoch_folding_curve(
    freq_grid: np.ndarray,
    toa_s: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int,
) -> np.ndarray:
    return np.array(
        [_epoch_folding_value(float(freq), toa_s, weights, n_bins=n_bins) for freq in freq_grid],
        dtype=np.float64,
    )


def _harmonic_power_curve(
    freq_grid: np.ndarray,
    toa_s: np.ndarray,
    weights: np.ndarray,
    *,
    n_harmonics: int,
    chunk_size: int = 256,
) -> np.ndarray:
    scores = np.empty(len(freq_grid), dtype=np.float64)
    denom = max(np.sum(weights), 1e-30)
    toa_row = toa_s[None, :]
    weights_row = weights[None, :]
    n_harmonics = max(int(n_harmonics), 1)
    for start in range(0, len(freq_grid), chunk_size):
        stop = min(start + chunk_size, len(freq_grid))
        freq_block = freq_grid[start:stop, None]
        base = np.exp(1j * 4.0 * np.pi * freq_block * toa_row)
        harmonic = base.copy()
        block_scores = np.zeros(stop - start, dtype=np.float64)
        for harmonic_idx in range(1, n_harmonics + 1):
            resultant = np.abs(np.sum(harmonic * weights_row, axis=1)) / denom
            block_scores += (resultant * resultant) / float(harmonic_idx)
            harmonic *= base
        scores[start:stop] = block_scores
    return scores


def _phase_distance_correlation_curve(
    freq_grid: np.ndarray,
    toa_s: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    values_centered, denom_right = _phase_distance_correlation_precompute(values)
    if denom_right <= 1e-30:
        return np.zeros(len(freq_grid), dtype=np.float64)

    toa = np.asarray(toa_s, dtype=np.float64)
    scores = np.zeros(len(freq_grid), dtype=np.float64)
    for idx, freq in enumerate(freq_grid):
        phases_unit = np.mod(toa * float(freq), 1.0)
        phase_diff = np.abs(phases_unit[:, None] - phases_unit[None, :])
        phase_diff = np.minimum(phase_diff, 1.0 - phase_diff)
        phase_dist = phase_diff * (1.0 - phase_diff)
        phase_centered = (
            phase_dist
            - phase_dist.mean(axis=0, keepdims=True)
            - phase_dist.mean(axis=1, keepdims=True)
            + phase_dist.mean()
        )
        numerator = float(np.sum(phase_centered * values_centered))
        denom_left = float(np.sum(phase_centered * phase_centered))
        denom = np.sqrt(max(denom_left * denom_right, 0.0))
        scores[idx] = float(max(numerator / denom, 0.0)) if denom > 1e-30 else 0.0
    return scores


def _score_value(
    freq_hz: float,
    toa_s: np.ndarray,
    weights: np.ndarray,
    *,
    method: str,
    n_harmonics: int,
) -> float:
    if method == "coherence":
        return _coherence_value(freq_hz, toa_s, weights)
    if method == "harmonic_power":
        return _harmonic_power_value(freq_hz, toa_s, weights, n_harmonics=n_harmonics)
    if method == "epoch_folding":
        return _epoch_folding_value(freq_hz, toa_s, weights, n_bins=24)
    if method == "gregory_loredo":
        return _gregory_loredo_value(freq_hz, toa_s)
    if method == "phase_distance_correlation":
        return _phase_distance_correlation_value(freq_hz, toa_s, weights)
    raise ValueError(f"Unknown blind PRPD frequency method: {method}")


def _score_curve(
    freq_grid: np.ndarray,
    toa_s: np.ndarray,
    weights: np.ndarray,
    *,
    method: str,
    n_harmonics: int,
    chunk_size: int = 256,
) -> np.ndarray:
    if method == "coherence":
        return _coherence_curve(freq_grid, toa_s, weights, chunk_size=chunk_size)
    if method == "harmonic_power":
        return _harmonic_power_curve(
            freq_grid,
            toa_s,
            weights,
            n_harmonics=n_harmonics,
            chunk_size=chunk_size,
        )
    if method == "epoch_folding":
        return _epoch_folding_curve(freq_grid, toa_s, weights, n_bins=24)
    if method == "gregory_loredo":
        return np.array([_gregory_loredo_value(float(freq), toa_s) for freq in freq_grid], dtype=np.float64)
    if method == "phase_distance_correlation":
        return _phase_distance_correlation_curve(freq_grid, toa_s, weights)
    raise ValueError(f"Unknown blind PRPD frequency method: {method}")


def _estimate_phase_offset_deg(
    toa_s: np.ndarray,
    *,
    freq_hz: float,
    weights: np.ndarray,
) -> float:
    doubled_phase = 4.0 * np.pi * float(freq_hz) * toa_s
    resultant = np.sum(weights * np.exp(1j * doubled_phase))
    if not np.isfinite(resultant.real) or not np.isfinite(resultant.imag):
        return 0.0
    return float(np.mod(np.rad2deg(np.angle(resultant)) / 2.0, 180.0))


def _phase_inlier_mask(phases_deg: np.ndarray, sigma_threshold: float = 2.5) -> np.ndarray:
    if len(phases_deg) < 20:
        return np.ones(len(phases_deg), dtype=bool)

    theta2 = np.deg2rad(phases_deg) * 2.0
    mean_angle = np.arctan2(np.mean(np.sin(theta2)), np.mean(np.cos(theta2))) / 2.0
    center1 = np.rad2deg(mean_angle) % 360.0
    center2 = (center1 + 180.0) % 360.0

    d1 = np.minimum(np.abs(phases_deg - center1), 360.0 - np.abs(phases_deg - center1))
    d2 = np.minimum(np.abs(phases_deg - center2), 360.0 - np.abs(phases_deg - center2))
    d_min = np.minimum(d1, d2)

    median_d = float(np.median(d_min))
    mad = float(np.median(np.abs(d_min - median_d)))
    threshold = median_d + sigma_threshold * max(mad * 1.4826, 5.0)
    return d_min <= threshold


def _axial_entropy_score(
    freq_hz: float,
    toa_s: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int = 24,
) -> float:
    phases_folded = np.mod(np.asarray(toa_s, dtype=np.float64) * float(freq_hz) * 360.0, 180.0)
    hist, _ = np.histogram(
        phases_folded,
        bins=max(int(n_bins), 8),
        range=(0.0, 180.0),
        weights=np.asarray(weights, dtype=np.float64),
    )
    total = float(np.sum(hist))
    if total <= 0.0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs))) / np.log(len(hist))
    return float(1.0 - entropy)


def _peak_shape_metrics(freq_grid: np.ndarray, scores: np.ndarray, best_idx: int) -> tuple[float, float, float]:
    if len(freq_grid) == 0 or len(scores) == 0 or best_idx < 0 or best_idx >= len(scores):
        return float("nan"), float("nan"), float("nan")

    peak_score = float(scores[best_idx])
    if not np.isfinite(peak_score):
        return float("nan"), float("nan"), float("nan")

    baseline = float(np.median(scores))
    prominence = max(peak_score - baseline, 0.0)
    if len(freq_grid) > 1:
        step_hz = float(np.median(np.diff(freq_grid)))
    else:
        step_hz = float("nan")

    width_hz = float("nan")
    if prominence > 0.0:
        threshold = peak_score - 0.5 * prominence
        left_idx = best_idx
        right_idx = best_idx
        while left_idx > 0 and float(scores[left_idx - 1]) >= threshold:
            left_idx -= 1
        while right_idx < len(scores) - 1 and float(scores[right_idx + 1]) >= threshold:
            right_idx += 1
        width_hz = float(freq_grid[right_idx] - freq_grid[left_idx])

    sharpness = float("nan")
    if np.isfinite(prominence) and np.isfinite(step_hz) and step_hz > 0.0:
        denom = width_hz if np.isfinite(width_hz) and width_hz > 0.0 else step_hz
        sharpness = float(prominence / max(denom, step_hz))
    return sharpness, width_hz, prominence


def _common_axial_support_metrics(
    freq_grid: np.ndarray,
    toa_s: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    if len(freq_grid) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    common_scores = np.array(
        [_axial_entropy_score(float(freq), toa_s, weights) for freq in freq_grid],
        dtype=np.float64,
    )
    common_best_idx = int(np.argmax(common_scores))
    common_peak_freq_hz = float(freq_grid[common_best_idx])
    common_sharpness, common_width_hz, common_prominence = _peak_shape_metrics(
        freq_grid,
        common_scores,
        common_best_idx,
    )
    span_hz = float(freq_grid[-1] - freq_grid[0]) if len(freq_grid) > 1 else float("nan")
    common_width_ratio = (
        float(common_width_hz / span_hz)
        if np.isfinite(common_width_hz) and np.isfinite(span_hz) and span_hz > 0.0
        else float("nan")
    )
    common_confidence = float("nan")
    if np.isfinite(common_prominence) and np.isfinite(common_width_ratio):
        common_confidence = float(
            common_prominence / max(common_prominence + common_width_ratio, 1e-12)
    )
    return (
        common_peak_freq_hz,
        common_sharpness,
        common_width_hz,
        common_prominence,
        common_width_ratio,
        common_confidence,
    )


def _bootstrap_stability_metrics(
    toa_s: np.ndarray,
    *,
    base_freq: float,
    search_width: float,
    coarse_steps: int,
    refine_half_width: float,
    max_events: int,
    peak_weights: np.ndarray | None,
    robust_refine: bool,
    method: str,
    n_harmonics: int,
    iterations: int,
    sample_fraction: float,
    seed: int | None,
    selected_method: str,
) -> dict[str, Any]:
    iterations = max(int(iterations), 0)
    if iterations <= 0:
        return {
            "bootstrap_iterations": 0,
            "bootstrap_sample_fraction": float("nan"),
            "bootstrap_freq_mean_hz": float("nan"),
            "bootstrap_freq_std_hz": float("nan"),
            "bootstrap_ci_low_hz": float("nan"),
            "bootstrap_ci_high_hz": float("nan"),
            "bootstrap_ci_width_hz": float("nan"),
            "bootstrap_method_agreement": float("nan"),
            "bootstrap_selected_method_counts": {},
        }

    toa = np.asarray(toa_s, dtype=np.float64)
    weights = None if peak_weights is None else np.asarray(peak_weights, dtype=np.float64)
    n_events = len(toa)
    if n_events < 20:
        return {
            "bootstrap_iterations": 0,
            "bootstrap_sample_fraction": float("nan"),
            "bootstrap_freq_mean_hz": float("nan"),
            "bootstrap_freq_std_hz": float("nan"),
            "bootstrap_ci_low_hz": float("nan"),
            "bootstrap_ci_high_hz": float("nan"),
            "bootstrap_ci_width_hz": float("nan"),
            "bootstrap_method_agreement": float("nan"),
            "bootstrap_selected_method_counts": {},
        }

    effective_fraction = min(max(float(sample_fraction), 0.5), 1.0)
    sample_size = min(n_events, max(20, int(round(n_events * effective_fraction))))
    rng = np.random.default_rng(seed)
    freq_samples: list[float] = []
    method_counts: dict[str, int] = {}
    bootstrap_coarse_steps = min(int(coarse_steps), 401)
    bootstrap_max_events = min(int(max_events), 4000)

    for _ in range(iterations):
        idx = np.sort(rng.integers(0, n_events, size=sample_size))
        boot_toa = toa[idx]
        boot_weights = weights[idx] if weights is not None else None
        boot_result = calibrate_grid_frequency_details(
            boot_toa,
            base_freq=base_freq,
            search_width=search_width,
            coarse_steps=bootstrap_coarse_steps,
            refine_half_width=refine_half_width,
            max_events=bootstrap_max_events,
            peak_weights=boot_weights,
            robust_refine=robust_refine,
            method=method,
            n_harmonics=n_harmonics,
            bootstrap_iterations=0,
        )
        freq_samples.append(float(boot_result.freq_hz))
        boot_method = str(boot_result.selected_method)
        method_counts[boot_method] = method_counts.get(boot_method, 0) + 1

    freq_arr = np.asarray(freq_samples, dtype=np.float64)
    ci_low_hz = float(np.quantile(freq_arr, 0.025))
    ci_high_hz = float(np.quantile(freq_arr, 0.975))
    method_agreement = float(method_counts.get(str(selected_method), 0) / max(iterations, 1))
    return {
        "bootstrap_iterations": int(iterations),
        "bootstrap_sample_fraction": float(effective_fraction),
        "bootstrap_freq_mean_hz": float(np.mean(freq_arr)),
        "bootstrap_freq_std_hz": float(np.std(freq_arr, ddof=0)),
        "bootstrap_ci_low_hz": ci_low_hz,
        "bootstrap_ci_high_hz": ci_high_hz,
        "bootstrap_ci_width_hz": float(ci_high_hz - ci_low_hz),
        "bootstrap_method_agreement": method_agreement,
        "bootstrap_selected_method_counts": method_counts,
    }


def _contiguous_window_stability_metrics(
    toa_s: np.ndarray,
    *,
    base_freq: float,
    search_width: float,
    coarse_steps: int,
    refine_half_width: float,
    max_events: int,
    peak_weights: np.ndarray | None,
    robust_refine: bool,
    method: str,
    n_harmonics: int,
    window_size_events: int,
    step_events: int,
    min_events_per_window: int,
    min_window_count: int,
    selected_method: str,
) -> dict[str, Any]:
    if int(window_size_events) <= 0 or int(step_events) <= 0:
        return {
            "local_window_count": 0,
            "local_window_size_events": 0,
            "local_window_step_events": 0,
            "local_freq_mean_hz": float("nan"),
            "local_freq_std_hz": float("nan"),
            "local_freq_min_hz": float("nan"),
            "local_freq_max_hz": float("nan"),
            "local_freq_span_hz": float("nan"),
            "local_method_agreement": float("nan"),
            "local_common_confidence_mean": float("nan"),
            "local_dominant_method": "",
            "local_selected_method_counts": {},
        }

    toa = np.asarray(toa_s, dtype=np.float64)
    weights = None if peak_weights is None else np.asarray(peak_weights, dtype=np.float64)
    n_events = len(toa)
    window_size = max(int(window_size_events), int(min_events_per_window))
    if n_events < window_size:
        return {
            "local_window_count": 0,
            "local_window_size_events": window_size,
            "local_window_step_events": int(step_events),
            "local_freq_mean_hz": float("nan"),
            "local_freq_std_hz": float("nan"),
            "local_freq_min_hz": float("nan"),
            "local_freq_max_hz": float("nan"),
            "local_freq_span_hz": float("nan"),
            "local_method_agreement": float("nan"),
            "local_common_confidence_mean": float("nan"),
            "local_dominant_method": "",
            "local_selected_method_counts": {},
        }

    freq_values: list[float] = []
    confidence_values: list[float] = []
    method_counts: dict[str, int] = {}
    effective_step = max(int(step_events), 1)
    window_coarse_steps = min(int(coarse_steps), 1001)
    window_max_events = min(int(max_events), max(window_size, 512))

    for start in range(0, n_events - window_size + 1, effective_step):
        end = start + window_size
        window_toa = toa[start:end]
        window_weights = weights[start:end] if weights is not None else None
        window_result = calibrate_grid_frequency_details(
            window_toa,
            base_freq=base_freq,
            search_width=search_width,
            coarse_steps=window_coarse_steps,
            refine_half_width=refine_half_width,
            max_events=window_max_events,
            peak_weights=window_weights,
            robust_refine=robust_refine,
            method=method,
            n_harmonics=n_harmonics,
            bootstrap_iterations=0,
            local_window_size_events=0,
        )
        freq_values.append(float(window_result.freq_hz))
        confidence_values.append(float(window_result.common_axial_confidence))
        method_key = str(window_result.selected_method)
        method_counts[method_key] = method_counts.get(method_key, 0) + 1

    window_count = len(freq_values)
    if window_count < max(int(min_window_count), 1):
        return {
            "local_window_count": int(window_count),
            "local_window_size_events": int(window_size),
            "local_window_step_events": int(effective_step),
            "local_freq_mean_hz": float("nan"),
            "local_freq_std_hz": float("nan"),
            "local_freq_min_hz": float("nan"),
            "local_freq_max_hz": float("nan"),
            "local_freq_span_hz": float("nan"),
            "local_method_agreement": float("nan"),
            "local_common_confidence_mean": float("nan"),
            "local_dominant_method": "",
            "local_selected_method_counts": method_counts,
        }

    freq_arr = np.asarray(freq_values, dtype=np.float64)
    conf_arr = np.asarray(confidence_values, dtype=np.float64)
    dominant_method = ""
    if method_counts:
        dominant_method = max(method_counts.items(), key=lambda item: (int(item[1]), str(item[0])))[0]
    return {
        "local_window_count": int(window_count),
        "local_window_size_events": int(window_size),
        "local_window_step_events": int(effective_step),
        "local_freq_mean_hz": float(np.mean(freq_arr)),
        "local_freq_std_hz": float(np.std(freq_arr, ddof=0)),
        "local_freq_min_hz": float(np.min(freq_arr)),
        "local_freq_max_hz": float(np.max(freq_arr)),
        "local_freq_span_hz": float(np.max(freq_arr) - np.min(freq_arr)),
        "local_method_agreement": float(method_counts.get(str(selected_method), 0) / max(window_count, 1)),
        "local_common_confidence_mean": float(np.nanmean(conf_arr)),
        "local_dominant_method": str(dominant_method),
        "local_selected_method_counts": method_counts,
    }


def compute_local_blind_prpd_trace(
    toa_s: np.ndarray,
    *,
    base_freq: float = 50.0,
    search_width: float = 0.5,
    coarse_steps: int = 2001,
    refine_half_width: float = 0.02,
    max_events: int = 20000,
    peak_weights: np.ndarray | None = None,
    robust_refine: bool = True,
    method: str = "auto",
    n_harmonics: int = 3,
    window_size_events: int = 256,
    step_events: int = 128,
    min_events_per_window: int = 128,
    min_window_count: int = 3,
    global_freq_hz: float | None = None,
) -> list[dict[str, Any]]:
    toa = np.asarray(toa_s, dtype=np.float64)
    valid = np.isfinite(toa)
    toa = toa[valid]
    weights = None
    if peak_weights is not None:
        weights_arr = np.asarray(peak_weights, dtype=np.float64)
        if len(weights_arr) == len(valid):
            weights = weights_arr[valid]
        elif len(weights_arr) == len(toa):
            weights = weights_arr

    if int(window_size_events) <= 0 or int(step_events) <= 0:
        return []

    n_events = len(toa)
    window_size = max(int(window_size_events), int(min_events_per_window))
    if n_events < window_size:
        return []

    effective_step = max(int(step_events), 1)
    window_coarse_steps = min(int(coarse_steps), 1001)
    window_max_events = min(int(max_events), max(window_size, 512))
    rows: list[dict[str, Any]] = []

    for start in range(0, n_events - window_size + 1, effective_step):
        end = start + window_size
        window_toa = toa[start:end]
        window_weights = weights[start:end] if weights is not None else None
        window_result = calibrate_grid_frequency_details(
            window_toa,
            base_freq=base_freq,
            search_width=search_width,
            coarse_steps=window_coarse_steps,
            refine_half_width=refine_half_width,
            max_events=window_max_events,
            peak_weights=window_weights,
            robust_refine=robust_refine,
            method=method,
            n_harmonics=n_harmonics,
            bootstrap_iterations=0,
            local_window_size_events=0,
        )
        rows.append(
            {
                "local_window_index": int(len(rows)),
                "event_start_idx": int(start),
                "event_end_idx": int(end - 1),
                "toa_start_s": float(window_toa[0]),
                "toa_end_s": float(window_toa[-1]),
                "toa_center_s": float(0.5 * (window_toa[0] + window_toa[-1])),
                "n_events": int(len(window_toa)),
                "requested_method": str(window_result.requested_method),
                "selected_method": str(window_result.selected_method),
                "freq_hz": float(window_result.freq_hz),
                "freq_offset_from_global_hz": (
                    float(window_result.freq_hz - float(global_freq_hz))
                    if global_freq_hz is not None and np.isfinite(global_freq_hz)
                    else float("nan")
                ),
                "coherence": float(window_result.coherence),
                "axial_entropy_score": float(window_result.axial_entropy_score),
                "common_axial_confidence": float(window_result.common_axial_confidence),
                "common_axial_peak_offset_hz": float(window_result.common_axial_peak_offset_hz),
                "sharpness": float(window_result.sharpness),
                "half_height_width_hz": float(window_result.half_height_width_hz),
            }
        )

    if len(rows) < max(int(min_window_count), 1):
        return []
    return rows


def _calibrate_single_method(
    toa_s: np.ndarray,
    *,
    base_freq: float,
    search_width: float,
    coarse_steps: int,
    refine_half_width: float,
    max_events: int,
    peak_weights: np.ndarray | None,
    robust_refine: bool,
    method: str,
    n_harmonics: int,
) -> BlindPRPDCandidate:
    toa = np.asarray(toa_s, dtype=np.float64)
    toa = toa[np.isfinite(toa)]
    if len(toa) < 10:
        return BlindPRPDCandidate(
            method=method,
            freq_hz=float(base_freq),
            score=float("nan"),
            coherence=float("nan"),
            axial_entropy_score=float("nan"),
            sharpness=float("nan"),
            half_height_width_hz=float("nan"),
            score_prominence=float("nan"),
        )

    order = np.argsort(toa)
    toa = toa[order]
    weights = _prepare_weights(peak_weights, len(toa))[order]
    effective_max_events = max_events
    effective_coarse_steps = coarse_steps
    effective_robust_refine = robust_refine
    local_grid_points = 401
    if method == "phase_distance_correlation":
        effective_max_events = min(int(max_events), 384)
        effective_coarse_steps = min(int(coarse_steps), 301)
        effective_robust_refine = False
        local_grid_points = 81
    if method == "gregory_loredo":
        effective_max_events = min(int(max_events), 1024)
        effective_coarse_steps = min(int(coarse_steps), 401)
        effective_robust_refine = False
        local_grid_points = 121
    toa, weights = _deterministic_subsample(toa, weights, max_events=effective_max_events)

    freq_grid = np.linspace(base_freq - search_width, base_freq + search_width, effective_coarse_steps)
    coarse_scores = _score_curve(
        freq_grid,
        toa,
        weights,
        method=method,
        n_harmonics=n_harmonics,
    )
    best_idx = int(np.argmax(coarse_scores))
    coarse_best = float(freq_grid[best_idx])

    left_idx = max(best_idx - 1, 0)
    right_idx = min(best_idx + 1, len(freq_grid) - 1)
    left_bound = max(float(freq_grid[left_idx]), coarse_best - refine_half_width)
    right_bound = min(float(freq_grid[right_idx]), coarse_best + refine_half_width)
    if left_bound == right_bound:
        left_bound = coarse_best - refine_half_width
        right_bound = coarse_best + refine_half_width

    objective = lambda f: -_score_value(
        float(f),
        toa,
        weights,
        method=method,
        n_harmonics=n_harmonics,
    )
    refined = minimize_scalar(objective, bounds=(left_bound, right_bound), method="bounded")
    best_freq = float(refined.x) if refined.success else coarse_best
    best_score = _score_value(
        best_freq,
        toa,
        weights,
        method=method,
        n_harmonics=n_harmonics,
    )

    if effective_robust_refine and len(toa) >= 20:
        phase_offset = _estimate_phase_offset_deg(toa, freq_hz=best_freq, weights=weights)
        phases_deg = np.mod(toa * best_freq * 360.0 - phase_offset, 360.0)
        mask = _phase_inlier_mask(phases_deg)
        if np.sum(mask) >= max(20, int(0.5 * len(mask))):
            toa_in = toa[mask]
            weights_in = weights[mask]
            objective_in = lambda f: -_score_value(
                float(f),
                toa_in,
                weights_in,
                method=method,
                n_harmonics=n_harmonics,
            )
            refined = minimize_scalar(objective_in, bounds=(left_bound, right_bound), method="bounded")
            candidate_freq = float(refined.x) if refined.success else best_freq
            candidate_score = _score_value(
                candidate_freq,
                toa,
                weights,
                method=method,
                n_harmonics=n_harmonics,
            )
            if candidate_score >= best_score:
                best_freq = candidate_freq
                best_score = candidate_score

    local_grid = np.linspace(best_freq - refine_half_width, best_freq + refine_half_width, local_grid_points)
    local_scores = _score_curve(
        local_grid,
        toa,
        weights,
        method=method,
        n_harmonics=n_harmonics,
    )
    local_best_idx = int(np.argmax(local_scores))
    sharpness, width_hz, prominence = _peak_shape_metrics(local_grid, local_scores, local_best_idx)
    (
        common_peak_freq_hz,
        common_axial_sharpness,
        common_axial_width_hz,
        common_axial_prominence,
        common_axial_width_ratio,
        common_axial_confidence,
    ) = _common_axial_support_metrics(local_grid, toa, weights)
    coherence = _coherence_value(best_freq, toa, weights)
    axial_entropy = _axial_entropy_score(best_freq, toa, weights)
    return BlindPRPDCandidate(
        method=method,
        freq_hz=float(best_freq),
        score=float(best_score),
        coherence=float(coherence),
        axial_entropy_score=float(axial_entropy),
        sharpness=float(sharpness),
        half_height_width_hz=float(width_hz),
        score_prominence=float(prominence),
        common_axial_peak_freq_hz=float(common_peak_freq_hz),
        common_axial_peak_offset_hz=float(abs(common_peak_freq_hz - best_freq))
        if np.isfinite(common_peak_freq_hz)
        else float("nan"),
        common_axial_sharpness=float(common_axial_sharpness),
        common_axial_half_height_width_hz=float(common_axial_width_hz),
        common_axial_prominence=float(common_axial_prominence),
        common_axial_width_ratio=float(common_axial_width_ratio),
        common_axial_confidence=float(common_axial_confidence),
    )


def calibrate_grid_frequency_details(
    toa_s: np.ndarray,
    *,
    base_freq: float = 50.0,
    search_width: float = 0.5,
    coarse_steps: int = 2001,
    refine_half_width: float = 0.02,
    max_events: int = 20000,
    peak_weights: np.ndarray | None = None,
    robust_refine: bool = True,
    method: str = "coherence",
    n_harmonics: int = 3,
    bootstrap_iterations: int = 0,
    bootstrap_sample_fraction: float = 0.75,
    bootstrap_seed: int | None = None,
    local_window_size_events: int = 0,
    local_window_step_events: int = 0,
    local_min_events_per_window: int = 128,
    local_min_window_count: int = 3,
) -> BlindPRPDCalibrationResult:
    toa = np.asarray(toa_s, dtype=np.float64)
    toa = toa[np.isfinite(toa)]
    if len(toa) < 10:
        return BlindPRPDCalibrationResult(
            requested_method=str(method),
            selected_method=str(method),
            freq_hz=float(base_freq),
            score=float("nan"),
            coherence=float("nan"),
            axial_entropy_score=float("nan"),
            sharpness=float("nan"),
            half_height_width_hz=float("nan"),
            score_prominence=float("nan"),
        )

    if method == "auto":
        candidates = tuple(
            _calibrate_single_method(
                toa,
                base_freq=base_freq,
                search_width=search_width,
                coarse_steps=coarse_steps,
                refine_half_width=refine_half_width,
                max_events=max_events,
                peak_weights=peak_weights,
                robust_refine=robust_refine,
                method=candidate_method,
                n_harmonics=n_harmonics,
            )
            for candidate_method in ("coherence", "harmonic_power", "epoch_folding")
        )
        sorted_candidates = sorted(
            candidates,
            key=lambda item: (item.axial_entropy_score, item.coherence),
            reverse=True,
        )
        winner = sorted_candidates[0]
        runner_up = sorted_candidates[1] if len(sorted_candidates) > 1 else None
        candidate_freqs = np.array([candidate.freq_hz for candidate in candidates], dtype=np.float64)
        candidate_spread = float(np.nanmax(candidate_freqs) - np.nanmin(candidate_freqs))
        winner_margin = (
            float(winner.axial_entropy_score - runner_up.axial_entropy_score)
            if runner_up is not None
            else float("nan")
        )
        bootstrap_metrics = _bootstrap_stability_metrics(
            toa,
            base_freq=base_freq,
            search_width=search_width,
            coarse_steps=coarse_steps,
            refine_half_width=refine_half_width,
            max_events=max_events,
            peak_weights=peak_weights,
            robust_refine=robust_refine,
            method="auto",
            n_harmonics=n_harmonics,
            iterations=bootstrap_iterations,
            sample_fraction=bootstrap_sample_fraction,
            seed=bootstrap_seed,
            selected_method=winner.method,
        )
        local_metrics = _contiguous_window_stability_metrics(
            toa,
            base_freq=base_freq,
            search_width=search_width,
            coarse_steps=coarse_steps,
            refine_half_width=refine_half_width,
            max_events=max_events,
            peak_weights=peak_weights,
            robust_refine=robust_refine,
            method="auto",
            n_harmonics=n_harmonics,
            window_size_events=local_window_size_events,
            step_events=local_window_step_events,
            min_events_per_window=local_min_events_per_window,
            min_window_count=local_min_window_count,
            selected_method=winner.method,
        )
        return BlindPRPDCalibrationResult(
            requested_method="auto",
            selected_method=winner.method,
            freq_hz=winner.freq_hz,
            score=winner.score,
            coherence=winner.coherence,
            axial_entropy_score=winner.axial_entropy_score,
            sharpness=winner.sharpness,
            half_height_width_hz=winner.half_height_width_hz,
            score_prominence=winner.score_prominence,
            common_axial_peak_freq_hz=winner.common_axial_peak_freq_hz,
            common_axial_peak_offset_hz=winner.common_axial_peak_offset_hz,
            common_axial_sharpness=winner.common_axial_sharpness,
            common_axial_half_height_width_hz=winner.common_axial_half_height_width_hz,
            common_axial_prominence=winner.common_axial_prominence,
            common_axial_width_ratio=winner.common_axial_width_ratio,
            common_axial_confidence=winner.common_axial_confidence,
            bootstrap_iterations=int(bootstrap_metrics["bootstrap_iterations"]),
            bootstrap_sample_fraction=float(bootstrap_metrics["bootstrap_sample_fraction"]),
            bootstrap_freq_mean_hz=float(bootstrap_metrics["bootstrap_freq_mean_hz"]),
            bootstrap_freq_std_hz=float(bootstrap_metrics["bootstrap_freq_std_hz"]),
            bootstrap_ci_low_hz=float(bootstrap_metrics["bootstrap_ci_low_hz"]),
            bootstrap_ci_high_hz=float(bootstrap_metrics["bootstrap_ci_high_hz"]),
            bootstrap_ci_width_hz=float(bootstrap_metrics["bootstrap_ci_width_hz"]),
            bootstrap_method_agreement=float(bootstrap_metrics["bootstrap_method_agreement"]),
            bootstrap_selected_method_counts=dict(bootstrap_metrics["bootstrap_selected_method_counts"]),
            local_window_count=int(local_metrics["local_window_count"]),
            local_window_size_events=int(local_metrics["local_window_size_events"]),
            local_window_step_events=int(local_metrics["local_window_step_events"]),
            local_freq_mean_hz=float(local_metrics["local_freq_mean_hz"]),
            local_freq_std_hz=float(local_metrics["local_freq_std_hz"]),
            local_freq_min_hz=float(local_metrics["local_freq_min_hz"]),
            local_freq_max_hz=float(local_metrics["local_freq_max_hz"]),
            local_freq_span_hz=float(local_metrics["local_freq_span_hz"]),
            local_method_agreement=float(local_metrics["local_method_agreement"]),
            local_common_confidence_mean=float(local_metrics["local_common_confidence_mean"]),
            local_dominant_method=str(local_metrics["local_dominant_method"]),
            local_selected_method_counts=dict(local_metrics["local_selected_method_counts"]),
            candidate_spread_hz=candidate_spread,
            winner_margin=winner_margin,
            candidate_methods=candidates,
        )

    candidate = _calibrate_single_method(
        toa,
        base_freq=base_freq,
        search_width=search_width,
        coarse_steps=coarse_steps,
        refine_half_width=refine_half_width,
        max_events=max_events,
        peak_weights=peak_weights,
        robust_refine=robust_refine,
        method=method,
        n_harmonics=n_harmonics,
    )
    bootstrap_metrics = _bootstrap_stability_metrics(
        toa,
        base_freq=base_freq,
        search_width=search_width,
        coarse_steps=coarse_steps,
        refine_half_width=refine_half_width,
        max_events=max_events,
        peak_weights=peak_weights,
        robust_refine=robust_refine,
        method=method,
        n_harmonics=n_harmonics,
        iterations=bootstrap_iterations,
        sample_fraction=bootstrap_sample_fraction,
        seed=bootstrap_seed,
        selected_method=candidate.method,
    )
    local_metrics = _contiguous_window_stability_metrics(
        toa,
        base_freq=base_freq,
        search_width=search_width,
        coarse_steps=coarse_steps,
        refine_half_width=refine_half_width,
        max_events=max_events,
        peak_weights=peak_weights,
        robust_refine=robust_refine,
        method=method,
        n_harmonics=n_harmonics,
        window_size_events=local_window_size_events,
        step_events=local_window_step_events,
        min_events_per_window=local_min_events_per_window,
        min_window_count=local_min_window_count,
        selected_method=candidate.method,
    )

    return BlindPRPDCalibrationResult(
        requested_method=str(method),
        selected_method=candidate.method,
        freq_hz=candidate.freq_hz,
        score=candidate.score,
        coherence=candidate.coherence,
        axial_entropy_score=candidate.axial_entropy_score,
        sharpness=candidate.sharpness,
        half_height_width_hz=candidate.half_height_width_hz,
        score_prominence=candidate.score_prominence,
        common_axial_peak_freq_hz=candidate.common_axial_peak_freq_hz,
        common_axial_peak_offset_hz=candidate.common_axial_peak_offset_hz,
        common_axial_sharpness=candidate.common_axial_sharpness,
        common_axial_half_height_width_hz=candidate.common_axial_half_height_width_hz,
        common_axial_prominence=candidate.common_axial_prominence,
        common_axial_width_ratio=candidate.common_axial_width_ratio,
        common_axial_confidence=candidate.common_axial_confidence,
        bootstrap_iterations=int(bootstrap_metrics["bootstrap_iterations"]),
        bootstrap_sample_fraction=float(bootstrap_metrics["bootstrap_sample_fraction"]),
        bootstrap_freq_mean_hz=float(bootstrap_metrics["bootstrap_freq_mean_hz"]),
        bootstrap_freq_std_hz=float(bootstrap_metrics["bootstrap_freq_std_hz"]),
        bootstrap_ci_low_hz=float(bootstrap_metrics["bootstrap_ci_low_hz"]),
        bootstrap_ci_high_hz=float(bootstrap_metrics["bootstrap_ci_high_hz"]),
        bootstrap_ci_width_hz=float(bootstrap_metrics["bootstrap_ci_width_hz"]),
        bootstrap_method_agreement=float(bootstrap_metrics["bootstrap_method_agreement"]),
        bootstrap_selected_method_counts=dict(bootstrap_metrics["bootstrap_selected_method_counts"]),
        local_window_count=int(local_metrics["local_window_count"]),
        local_window_size_events=int(local_metrics["local_window_size_events"]),
        local_window_step_events=int(local_metrics["local_window_step_events"]),
        local_freq_mean_hz=float(local_metrics["local_freq_mean_hz"]),
        local_freq_std_hz=float(local_metrics["local_freq_std_hz"]),
        local_freq_min_hz=float(local_metrics["local_freq_min_hz"]),
        local_freq_max_hz=float(local_metrics["local_freq_max_hz"]),
        local_freq_span_hz=float(local_metrics["local_freq_span_hz"]),
        local_method_agreement=float(local_metrics["local_method_agreement"]),
        local_common_confidence_mean=float(local_metrics["local_common_confidence_mean"]),
        local_dominant_method=str(local_metrics["local_dominant_method"]),
        local_selected_method_counts=dict(local_metrics["local_selected_method_counts"]),
        candidate_methods=(candidate,),
    )


def calibrate_grid_frequency(
    toa_s: np.ndarray,
    *,
    base_freq: float = 50.0,
    search_width: float = 0.5,
    coarse_steps: int = 2001,
    refine_half_width: float = 0.02,
    max_events: int = 20000,
    peak_weights: np.ndarray | None = None,
    robust_refine: bool = True,
    method: str = "coherence",
    n_harmonics: int = 3,
    bootstrap_iterations: int = 0,
    bootstrap_sample_fraction: float = 0.75,
    bootstrap_seed: int | None = None,
    local_window_size_events: int = 0,
    local_window_step_events: int = 0,
    local_min_events_per_window: int = 128,
    local_min_window_count: int = 3,
    return_score: bool = False,
) -> float | tuple[float, float]:
    """
    Estimate the blind grid frequency using a weighted circular concentration
    objective on doubled phase, which is appropriate for bipolar PRPD symmetry.
    """
    details = calibrate_grid_frequency_details(
        toa_s,
        base_freq=base_freq,
        search_width=search_width,
        coarse_steps=coarse_steps,
        refine_half_width=refine_half_width,
        max_events=max_events,
        peak_weights=peak_weights,
        robust_refine=robust_refine,
        method=method,
        n_harmonics=n_harmonics,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_sample_fraction=bootstrap_sample_fraction,
        bootstrap_seed=bootstrap_seed,
        local_window_size_events=local_window_size_events,
        local_window_step_events=local_window_step_events,
        local_min_events_per_window=local_min_events_per_window,
        local_min_window_count=local_min_window_count,
    )
    if return_score:
        return float(details.freq_hz), float(details.score)
    return float(details.freq_hz)


def reconstruct_blind_prpd(
    toa_s: np.ndarray,
    peaks: np.ndarray,
    *,
    freq_hz: float = 50.0,
    auto_calibrate: bool = True,
    calibration_method: str = "auto",
    calibration_search_width_hz: float = 0.5,
    calibration_coarse_steps: int = 2001,
    calibration_refine_half_width_hz: float = 0.02,
    calibration_max_events: int = 20000,
    calibration_robust_refine: bool = True,
    n_harmonics: int = 3,
    bootstrap_iterations: int = 0,
    bootstrap_sample_fraction: float = 0.75,
    bootstrap_seed: int | None = None,
    local_window_size_events: int = 0,
    local_window_step_events: int = 0,
    local_min_events_per_window: int = 128,
    local_min_window_count: int = 3,
    display_center_deg: float | None = None,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, BlindPRPDInfo]:
    """
    Reconstruct a blind PRPD phase pattern without forcing an arbitrary 70 deg
    alignment. The scientific phase is aligned to the latent bipolar axis; an
    optional display shift can be added later for visualization only.
    """
    toa = np.asarray(toa_s, dtype=np.float64)
    peaks_arr = np.asarray(peaks, dtype=np.float64)
    valid = np.isfinite(toa) & np.isfinite(peaks_arr)
    toa = toa[valid]
    peaks_arr = peaks_arr[valid]
    if len(toa) == 0:
        empty = (np.array([]), np.array([]))
        if return_metadata:
            return empty[0], empty[1], BlindPRPDInfo(freq_hz, float("nan"), 0.0, 0.0, 0)
        return empty

    weights = _prepare_weights(peaks_arr, len(toa))
    details = BlindPRPDCalibrationResult(
        requested_method=str(calibration_method),
        selected_method=str(calibration_method),
        freq_hz=float(freq_hz),
        score=float("nan"),
        coherence=float("nan"),
        axial_entropy_score=float("nan"),
        sharpness=float("nan"),
        half_height_width_hz=float("nan"),
        score_prominence=float("nan"),
    )
    if auto_calibrate:
        details = calibrate_grid_frequency_details(
            toa,
            base_freq=freq_hz,
            search_width=calibration_search_width_hz,
            coarse_steps=calibration_coarse_steps,
            refine_half_width=calibration_refine_half_width_hz,
            max_events=calibration_max_events,
            peak_weights=peaks_arr,
            robust_refine=calibration_robust_refine,
            method=calibration_method,
            n_harmonics=n_harmonics,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_sample_fraction=bootstrap_sample_fraction,
            bootstrap_seed=bootstrap_seed,
            local_window_size_events=local_window_size_events,
            local_window_step_events=local_window_step_events,
            local_min_events_per_window=local_min_events_per_window,
            local_min_window_count=local_min_window_count,
        )
        print(
            f"[{__name__}] Frecuencia ciega calibrada: {details.freq_hz:.6f} Hz "
            f"(Base: {freq_hz} Hz, metodo pedido: {calibration_method}, ganador: {details.selected_method})"
        )
        freq_to_use = details.freq_hz
    else:
        freq_to_use = float(freq_hz)

    phase_offset_deg = _estimate_phase_offset_deg(toa, freq_hz=freq_to_use, weights=weights)
    raw_phase_deg = np.mod(toa * freq_to_use * 360.0, 360.0)
    scientific_phase_deg = np.mod(raw_phase_deg - phase_offset_deg, 360.0)

    display_shift_deg = 0.0
    if display_center_deg is not None:
        display_shift_deg = float(display_center_deg)
    phases_out = np.mod(scientific_phase_deg + display_shift_deg, 360.0)

    info = BlindPRPDInfo(
        calibrated_freq_hz=float(freq_to_use),
        coherence=float(details.coherence),
        phase_offset_deg=float(phase_offset_deg),
        display_shift_deg=float(display_shift_deg),
        n_events_used=int(len(toa)),
        method=str(details.selected_method),
        requested_method=str(details.requested_method),
        selected_method=str(details.selected_method),
        score=float(details.score),
        axial_entropy_score=float(details.axial_entropy_score),
        sharpness=float(details.sharpness),
        half_height_width_hz=float(details.half_height_width_hz),
        score_prominence=float(details.score_prominence),
        common_axial_peak_freq_hz=float(details.common_axial_peak_freq_hz),
        common_axial_peak_offset_hz=float(details.common_axial_peak_offset_hz),
        common_axial_sharpness=float(details.common_axial_sharpness),
        common_axial_half_height_width_hz=float(details.common_axial_half_height_width_hz),
        common_axial_prominence=float(details.common_axial_prominence),
        common_axial_width_ratio=float(details.common_axial_width_ratio),
        common_axial_confidence=float(details.common_axial_confidence),
        bootstrap_iterations=int(details.bootstrap_iterations),
        bootstrap_sample_fraction=float(details.bootstrap_sample_fraction),
        bootstrap_freq_mean_hz=float(details.bootstrap_freq_mean_hz),
        bootstrap_freq_std_hz=float(details.bootstrap_freq_std_hz),
        bootstrap_ci_low_hz=float(details.bootstrap_ci_low_hz),
        bootstrap_ci_high_hz=float(details.bootstrap_ci_high_hz),
        bootstrap_ci_width_hz=float(details.bootstrap_ci_width_hz),
        bootstrap_method_agreement=float(details.bootstrap_method_agreement),
        bootstrap_selected_method_counts=dict(details.bootstrap_selected_method_counts),
        local_window_count=int(details.local_window_count),
        local_window_size_events=int(details.local_window_size_events),
        local_window_step_events=int(details.local_window_step_events),
        local_freq_mean_hz=float(details.local_freq_mean_hz),
        local_freq_std_hz=float(details.local_freq_std_hz),
        local_freq_min_hz=float(details.local_freq_min_hz),
        local_freq_max_hz=float(details.local_freq_max_hz),
        local_freq_span_hz=float(details.local_freq_span_hz),
        local_method_agreement=float(details.local_method_agreement),
        local_common_confidence_mean=float(details.local_common_confidence_mean),
        local_dominant_method=str(details.local_dominant_method),
        local_selected_method_counts=dict(details.local_selected_method_counts),
        candidate_spread_hz=float(details.candidate_spread_hz),
        winner_margin=float(details.winner_margin),
        candidate_methods=details.candidate_methods,
    )
    if return_metadata:
        return phases_out, peaks_arr, info
    return phases_out, peaks_arr


def compare_frequency_estimators(
    toa_s: np.ndarray,
    *,
    base_freq: float = 50.0,
    search_width: float = 0.5,
    coarse_steps: int = 2001,
    refine_half_width: float = 0.02,
    max_events: int = 20000,
    peak_weights: np.ndarray | None = None,
    methods: tuple[str, ...] = ("coherence", "harmonic_power", "epoch_folding"),
    n_harmonics: int = 3,
    bootstrap_iterations: int = 0,
    bootstrap_sample_fraction: float = 0.75,
    bootstrap_seed: int | None = None,
    local_window_size_events: int = 0,
    local_window_step_events: int = 0,
    local_min_events_per_window: int = 128,
    local_min_window_count: int = 3,
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for method in methods:
        details = calibrate_grid_frequency_details(
            toa_s,
            base_freq=base_freq,
            search_width=search_width,
            coarse_steps=coarse_steps,
            refine_half_width=refine_half_width,
            max_events=max_events,
            peak_weights=peak_weights,
            method=method,
            n_harmonics=n_harmonics,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_sample_fraction=bootstrap_sample_fraction,
            bootstrap_seed=bootstrap_seed,
            local_window_size_events=local_window_size_events,
            local_window_step_events=local_window_step_events,
            local_min_events_per_window=local_min_events_per_window,
            local_min_window_count=local_min_window_count,
        )
        results.append(
            {
                "method": method,
                "selected_method": str(details.selected_method),
                "freq_hz": float(details.freq_hz),
                "score": float(details.score),
                "coherence": float(details.coherence),
                "axial_entropy_score": float(details.axial_entropy_score),
                "sharpness": float(details.sharpness),
                "half_height_width_hz": float(details.half_height_width_hz),
                "score_prominence": float(details.score_prominence),
                "common_axial_peak_freq_hz": float(details.common_axial_peak_freq_hz),
                "common_axial_peak_offset_hz": float(details.common_axial_peak_offset_hz),
                "common_axial_sharpness": float(details.common_axial_sharpness),
                "common_axial_half_height_width_hz": float(details.common_axial_half_height_width_hz),
                "common_axial_prominence": float(details.common_axial_prominence),
                "common_axial_width_ratio": float(details.common_axial_width_ratio),
                "common_axial_confidence": float(details.common_axial_confidence),
                "bootstrap_iterations": int(details.bootstrap_iterations),
                "bootstrap_sample_fraction": float(details.bootstrap_sample_fraction),
                "bootstrap_freq_mean_hz": float(details.bootstrap_freq_mean_hz),
                "bootstrap_freq_std_hz": float(details.bootstrap_freq_std_hz),
                "bootstrap_ci_low_hz": float(details.bootstrap_ci_low_hz),
                "bootstrap_ci_high_hz": float(details.bootstrap_ci_high_hz),
                "bootstrap_ci_width_hz": float(details.bootstrap_ci_width_hz),
                "bootstrap_method_agreement": float(details.bootstrap_method_agreement),
                "local_window_count": int(details.local_window_count),
                "local_window_size_events": int(details.local_window_size_events),
                "local_window_step_events": int(details.local_window_step_events),
                "local_freq_mean_hz": float(details.local_freq_mean_hz),
                "local_freq_std_hz": float(details.local_freq_std_hz),
                "local_freq_min_hz": float(details.local_freq_min_hz),
                "local_freq_max_hz": float(details.local_freq_max_hz),
                "local_freq_span_hz": float(details.local_freq_span_hz),
                "local_method_agreement": float(details.local_method_agreement),
                "local_common_confidence_mean": float(details.local_common_confidence_mean),
                "local_dominant_method": str(details.local_dominant_method),
                "candidate_spread_hz": float(details.candidate_spread_hz),
                "winner_margin": float(details.winner_margin),
            }
        )
    return results
