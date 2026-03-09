from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deltapd.blind_prpd import compare_frequency_estimators


METHODS = ("coherence", "harmonic_power", "epoch_folding", "gregory_loredo", "auto")


def _benchmark_output_dir() -> Path:
    return Path("outputs") / "blind_prpd_stress"


def _build_cycle_starts(freq_per_cycle_hz: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq_per_cycle_hz, dtype=np.float64)
    starts = np.zeros(len(freq), dtype=np.float64)
    if len(freq) > 1:
        starts[1:] = np.cumsum(1.0 / np.maximum(freq[:-1], 1e-12))
    return starts


def _apply_segment_gaps(
    cycle_starts_s: np.ndarray,
    *,
    rng: np.random.Generator,
    n_segments: int,
    gap_min_s: float,
    gap_max_s: float,
) -> tuple[np.ndarray, int, float, float]:
    starts = np.asarray(cycle_starts_s, dtype=np.float64).copy()
    n_cycles = len(starts)
    if n_cycles == 0 or n_segments <= 1:
        return starts, 0, 0.0, 0.0

    edges = np.linspace(0, n_cycles, n_segments + 1, dtype=int)
    gaps_s = rng.uniform(gap_min_s, gap_max_s, size=n_segments - 1)
    cumulative_gap = 0.0
    for seg_idx in range(1, n_segments):
        cumulative_gap += float(gaps_s[seg_idx - 1])
        starts[edges[seg_idx] :] += cumulative_gap
    return starts, int(len(gaps_s)), float(np.sum(gaps_s)), float(np.max(gaps_s))


def _weighted_reference_frequency_hz(
    instantaneous_freq_hz: np.ndarray,
    peaks: np.ndarray,
) -> float:
    weights = np.sqrt(np.abs(np.asarray(peaks, dtype=np.float64)))
    weights[~np.isfinite(weights)] = 0.0
    freq = np.asarray(instantaneous_freq_hz, dtype=np.float64)
    mask = np.isfinite(freq)
    if not np.any(mask):
        return float("nan")
    safe_weights = weights[mask]
    if float(np.sum(safe_weights)) <= 0.0:
        safe_weights = np.ones(np.sum(mask), dtype=np.float64)
    return float(np.average(freq[mask], weights=safe_weights))


def _optimal_axial_shift_deg(
    estimated_raw_phase_deg: np.ndarray,
    true_phase_deg: np.ndarray,
    peaks: np.ndarray,
) -> float:
    est = np.asarray(estimated_raw_phase_deg, dtype=np.float64)
    truth = np.asarray(true_phase_deg, dtype=np.float64)
    weights = np.sqrt(np.abs(np.asarray(peaks, dtype=np.float64)))
    mask = np.isfinite(est) & np.isfinite(truth) & np.isfinite(weights)
    if np.count_nonzero(mask) == 0:
        return 0.0
    doubled_delta = np.deg2rad(2.0 * (est[mask] - truth[mask]))
    resultant = np.sum(weights[mask] * np.exp(1j * doubled_delta))
    if not np.isfinite(resultant.real) or not np.isfinite(resultant.imag):
        return 0.0
    return float(np.mod(np.rad2deg(np.angle(resultant)) / 2.0, 180.0))


def _axial_phase_error_summary(
    *,
    toa_s: np.ndarray,
    true_phase_deg: np.ndarray,
    peaks: np.ndarray,
    estimated_freq_hz: float,
) -> dict[str, float]:
    raw_phase_deg = np.mod(np.asarray(toa_s, dtype=np.float64) * float(estimated_freq_hz) * 360.0, 360.0)
    true_phase = np.asarray(true_phase_deg, dtype=np.float64)
    shift_deg = _optimal_axial_shift_deg(raw_phase_deg, true_phase, peaks)
    aligned_phase_deg = np.mod(raw_phase_deg - shift_deg, 360.0)
    doubled_error = np.angle(np.exp(1j * np.deg2rad(2.0 * (aligned_phase_deg - true_phase))))
    axial_abs_error_deg = np.abs(np.rad2deg(doubled_error)) / 2.0
    return {
        "phase_shift_alignment_deg": float(shift_deg),
        "mean_axial_phase_error_deg": float(np.mean(axial_abs_error_deg)),
        "median_axial_phase_error_deg": float(np.median(axial_abs_error_deg)),
        "p95_axial_phase_error_deg": float(np.quantile(axial_abs_error_deg, 0.95)),
    }


def _scenario_event_stream(
    scenario: str,
    *,
    rng: np.random.Generator,
    base_freq_hz: float,
    n_cycles: int,
) -> dict[str, np.ndarray | float | int | str]:
    cycles = np.arange(n_cycles, dtype=np.float64)
    phase_choices = rng.choice([35.0, 215.0], size=n_cycles)
    jitter_s = rng.normal(0.0, 2.5e-6, size=n_cycles)
    peaks = 0.7 + 0.5 * rng.random(n_cycles)
    gap_count = 0
    total_gap_s = 0.0
    max_gap_s = 0.0

    if scenario == "linear_drift_mild":
        freq_per_cycle = np.linspace(base_freq_hz - 0.06, base_freq_hz + 0.06, n_cycles, dtype=np.float64)
    elif scenario == "linear_drift_strong":
        freq_per_cycle = np.linspace(base_freq_hz - 0.20, base_freq_hz + 0.20, n_cycles, dtype=np.float64)
    elif scenario == "segmented_gaps":
        freq_per_cycle = np.full(n_cycles, base_freq_hz, dtype=np.float64)
    elif scenario == "drift_segmented_gaps":
        freq_per_cycle = np.linspace(base_freq_hz - 0.14, base_freq_hz + 0.14, n_cycles, dtype=np.float64)
        peaks = np.where(phase_choices < 180.0, 1.0 + 0.25 * rng.random(n_cycles), 0.45 + 0.20 * rng.random(n_cycles))
    elif scenario == "drift_gaps_dropout":
        freq_per_cycle = np.linspace(base_freq_hz - 0.18, base_freq_hz + 0.18, n_cycles, dtype=np.float64)
        peaks = np.where(phase_choices < 180.0, 1.05 + 0.20 * rng.random(n_cycles), 0.50 + 0.16 * rng.random(n_cycles))
    else:
        raise ValueError(f"Unknown stress scenario: {scenario}")

    cycle_starts_s = _build_cycle_starts(freq_per_cycle)
    if scenario in {"segmented_gaps", "drift_segmented_gaps", "drift_gaps_dropout"}:
        cycle_starts_s, gap_count, total_gap_s, max_gap_s = _apply_segment_gaps(
            cycle_starts_s,
            rng=rng,
            n_segments=6 if scenario != "segmented_gaps" else 5,
            gap_min_s=0.06,
            gap_max_s=0.24,
        )

    toa_s = cycle_starts_s + (phase_choices / 360.0) / np.maximum(freq_per_cycle, 1e-12) + jitter_s
    instantaneous_freq_hz = freq_per_cycle.copy()

    if scenario == "drift_gaps_dropout":
        keep_mask = rng.random(n_cycles) > 0.24
        toa_s = toa_s[keep_mask]
        phase_choices = phase_choices[keep_mask]
        peaks = peaks[keep_mask]
        instantaneous_freq_hz = instantaneous_freq_hz[keep_mask]

    order = np.argsort(toa_s)
    toa_s = np.asarray(toa_s[order], dtype=np.float64)
    phase_choices = np.asarray(phase_choices[order], dtype=np.float64)
    peaks = np.asarray(peaks[order], dtype=np.float64)
    instantaneous_freq_hz = np.asarray(instantaneous_freq_hz[order], dtype=np.float64)
    return {
        "toa_s": toa_s,
        "true_phase_deg": phase_choices,
        "peaks": peaks,
        "instantaneous_freq_hz": instantaneous_freq_hz,
        "drift_span_hz": float(np.max(instantaneous_freq_hz) - np.min(instantaneous_freq_hz)) if len(instantaneous_freq_hz) else 0.0,
        "gap_count": gap_count,
        "total_gap_s": total_gap_s,
        "max_gap_s": max_gap_s,
    }


def _write_heatmap(
    summary_df: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis_r",
) -> Path:
    pivot = summary_df.pivot(index="scenario", columns="method", values=value_column)
    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=list(pivot.columns), rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=list(pivot.index))
    for row_idx, scenario in enumerate(pivot.index):
        for col_idx, method in enumerate(pivot.columns):
            value = float(pivot.loc[scenario, method])
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def run_blind_prpd_stress_benchmark(
    *,
    output_dir: str | Path,
    methods: tuple[str, ...] = METHODS,
    scenarios: tuple[str, ...] = (
        "linear_drift_mild",
        "linear_drift_strong",
        "segmented_gaps",
        "drift_segmented_gaps",
        "drift_gaps_dropout",
    ),
    n_trials: int = 12,
    seed: int = 142,
    cycle_range: tuple[int, int] = (1200, 2200),
    search_width: float = 0.5,
    coarse_steps: int = 1001,
    max_events: int = 8000,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for scenario in scenarios:
        for trial_idx in range(n_trials):
            base_freq_hz = 50.0 + rng.uniform(-0.22, 0.22)
            n_cycles = int(rng.integers(cycle_range[0], cycle_range[1] + 1))
            payload = _scenario_event_stream(
                scenario,
                rng=rng,
                base_freq_hz=base_freq_hz,
                n_cycles=n_cycles,
            )
            toa_s = np.asarray(payload["toa_s"], dtype=np.float64)
            peaks = np.asarray(payload["peaks"], dtype=np.float64)
            true_phase_deg = np.asarray(payload["true_phase_deg"], dtype=np.float64)
            instantaneous_freq_hz = np.asarray(payload["instantaneous_freq_hz"], dtype=np.float64)
            reference_freq_hz = _weighted_reference_frequency_hz(instantaneous_freq_hz, peaks)

            results = compare_frequency_estimators(
                toa_s,
                base_freq=50.0,
                search_width=search_width,
                coarse_steps=coarse_steps,
                refine_half_width=0.02,
                max_events=max_events,
                peak_weights=peaks,
                methods=methods,
                n_harmonics=4,
            )
            for result in results:
                estimated_freq_hz = float(result["freq_hz"])
                phase_metrics = _axial_phase_error_summary(
                    toa_s=toa_s,
                    true_phase_deg=true_phase_deg,
                    peaks=peaks,
                    estimated_freq_hz=estimated_freq_hz,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "trial_idx": trial_idx,
                        "method": str(result["method"]),
                        "selected_method": str(result.get("selected_method", result["method"])),
                        "f_reference_hz": reference_freq_hz,
                        "f_est_hz": estimated_freq_hz,
                        "abs_error_vs_reference_hz": abs(estimated_freq_hz - reference_freq_hz),
                        "score": float(result["score"]),
                        "coherence": float(result["coherence"]),
                        "axial_entropy_score": float(result["axial_entropy_score"]),
                        "sharpness": float(result["sharpness"]),
                        "half_height_width_hz": float(result["half_height_width_hz"]),
                        "score_prominence": float(result["score_prominence"]),
                        "common_axial_confidence": float(result.get("common_axial_confidence", float("nan"))),
                        "common_axial_peak_offset_hz": float(result.get("common_axial_peak_offset_hz", float("nan"))),
                        "candidate_spread_hz": float(result["candidate_spread_hz"]),
                        "winner_margin": float(result["winner_margin"]),
                        "drift_span_hz": float(payload["drift_span_hz"]),
                        "gap_count": int(payload["gap_count"]),
                        "total_gap_s": float(payload["total_gap_s"]),
                        "max_gap_s": float(payload["max_gap_s"]),
                        "n_events": int(len(toa_s)),
                        **phase_metrics,
                    }
                )

    detail_df = pd.DataFrame(rows)
    detail_csv = output_dir / "blind_prpd_stress_detail.csv"
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    summary_df = (
        detail_df.groupby(["scenario", "method"], as_index=False)
        .agg(
            trials=("trial_idx", "count"),
            mean_abs_error_vs_reference_hz=("abs_error_vs_reference_hz", "mean"),
            p95_abs_error_vs_reference_hz=("abs_error_vs_reference_hz", lambda s: float(np.quantile(s, 0.95))),
            mean_axial_phase_error_deg=("mean_axial_phase_error_deg", "mean"),
            p95_axial_phase_error_deg=("p95_axial_phase_error_deg", "mean"),
            mean_sharpness=("sharpness", "mean"),
            mean_coherence=("coherence", "mean"),
            mean_axial_entropy_score=("axial_entropy_score", "mean"),
            mean_common_axial_confidence=("common_axial_confidence", "mean"),
            mean_common_axial_peak_offset_hz=("common_axial_peak_offset_hz", "mean"),
            mean_drift_span_hz=("drift_span_hz", "mean"),
            mean_total_gap_s=("total_gap_s", "mean"),
            mean_max_gap_s=("max_gap_s", "mean"),
        )
        .sort_values(["scenario", "mean_axial_phase_error_deg", "mean_abs_error_vs_reference_hz", "method"])
        .reset_index(drop=True)
    )
    summary_csv = output_dir / "blind_prpd_stress_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    winner_df = (
        summary_df.sort_values(
            ["scenario", "mean_axial_phase_error_deg", "mean_abs_error_vs_reference_hz", "method"]
        )
        .groupby("scenario", as_index=False)
        .first()
    )

    phase_heatmap = _write_heatmap(
        summary_df,
        value_column="mean_axial_phase_error_deg",
        title="Blind PRPD stress: mean axial phase error (deg)",
        out_path=output_dir / "blind_prpd_stress_phase_error_heatmap.png",
    )
    freq_heatmap = _write_heatmap(
        summary_df,
        value_column="mean_abs_error_vs_reference_hz",
        title="Blind PRPD stress: mean frequency error vs reference (Hz)",
        out_path=output_dir / "blind_prpd_stress_frequency_error_heatmap.png",
    )

    lines = [
        "# Blind PRPD stress benchmark",
        "",
        "## Why this benchmark is different",
        "",
        "- Here the main score is not only frequency error.",
        "- Under drift there is no single perfect constant frequency, so the benchmark also measures axial phase error after optimal global alignment against the known generating phases.",
        "- The frequency reference is the event-weighted mean instantaneous frequency of the synthetic stream.",
        "- `sharpness` is still exported, but it should be read within each method family only because the score scale changes across objectives.",
        "",
        "## Scenarios",
        "",
        "- `linear_drift_mild`: slow linear drift with no macro-gaps.",
        "- `linear_drift_strong`: stronger linear drift with no macro-gaps.",
        "- `segmented_gaps`: constant frequency with acquisition-like macro-gaps between segments.",
        "- `drift_segmented_gaps`: linear drift plus macro-gaps.",
        "- `drift_gaps_dropout`: drift plus macro-gaps plus event dropout.",
        "",
        "## Winners by scenario",
        "",
        "| Scenario | Winner | Mean axial phase error (deg) | Mean freq error vs ref (Hz) | Mean drift span (Hz) | Mean total gap (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in winner_df.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['mean_axial_phase_error_deg']:.4f} | "
            f"{row['mean_abs_error_vs_reference_hz']:.6f} | {row['mean_drift_span_hz']:.4f} | {row['mean_total_gap_s']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Full summary",
            "",
            "| Scenario | Method | Mean axial phase error (deg) | P95 axial phase error (deg) | Mean freq error vs ref (Hz) | P95 freq error vs ref (Hz) | Mean common conf. | Mean peak offset (Hz) | Mean axial entropy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(row["method"]),
                    f"{row['mean_axial_phase_error_deg']:.4f}",
                    f"{row['p95_axial_phase_error_deg']:.4f}",
                    f"{row['mean_abs_error_vs_reference_hz']:.6f}",
                    f"{row['p95_abs_error_vs_reference_hz']:.6f}",
                    f"{row['mean_common_axial_confidence']:.4f}",
                    f"{row['mean_common_axial_peak_offset_hz']:.6f}",
                    f"{row['mean_axial_entropy_score']:.4f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Output figures",
            "",
            f"- Phase error heatmap: `{phase_heatmap.as_posix()}`",
            f"- Frequency error heatmap: `{freq_heatmap.as_posix()}`",
            "",
        ]
    )

    summary_md = output_dir / "blind_prpd_stress_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "phase_heatmap": phase_heatmap,
        "frequency_heatmap": freq_heatmap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run drift/gap stress tests for blind PRPD methods.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_benchmark_output_dir(),
        help="Directory where stress-benchmark outputs will be written.",
    )
    parser.add_argument("--trials", type=int, default=12, help="Number of trials per scenario.")
    parser.add_argument("--seed", type=int, default=142, help="Random seed.")
    args = parser.parse_args()

    outputs = run_blind_prpd_stress_benchmark(
        output_dir=args.output_dir,
        n_trials=args.trials,
        seed=args.seed,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
