from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from deltapd.blind_prpd import compare_frequency_estimators


def _repo_root() -> Path:
    return _REPO_ROOT


def _scenario_toa(
    scenario: str,
    *,
    rng: np.random.Generator,
    f_true: float,
    n_cycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    cycles = np.arange(n_cycles, dtype=np.float64)

    if scenario == "symmetric_moderate":
        phases_deg = rng.choice([35.0, 215.0], size=n_cycles)
        jitter_s = rng.normal(0.0, 3.0e-6, size=n_cycles)
        toa = cycles / f_true + (phases_deg / 360.0) / f_true + jitter_s
        peaks = 0.7 + 0.6 * rng.random(n_cycles)
        return np.sort(toa), peaks[np.argsort(toa)]

    if scenario == "narrow_outliers":
        phases_deg = rng.choice([18.0, 198.0], size=n_cycles)
        jitter_s = rng.normal(0.0, 1.3e-6, size=n_cycles)
        toa_core = cycles / f_true + (phases_deg / 360.0) / f_true + jitter_s
        peaks_core = 0.9 + 0.3 * rng.random(n_cycles)

        n_outliers = max(40, n_cycles // 18)
        toa_out = rng.uniform(0.0, cycles.max() / f_true, size=n_outliers)
        peaks_out = 0.15 + 0.10 * rng.random(n_outliers)

        toa = np.concatenate([toa_core, toa_out])
        peaks = np.concatenate([peaks_core, peaks_out])
        order = np.argsort(toa)
        return toa[order], peaks[order]

    if scenario == "asymmetric_polarity":
        phase_options = np.array([28.0, 208.0], dtype=np.float64)
        polarity = rng.choice([0, 1], p=[0.72, 0.28], size=n_cycles)
        phases_deg = phase_options[polarity]
        jitter_s = rng.normal(0.0, 2.1e-6, size=n_cycles)
        toa = cycles / f_true + (phases_deg / 360.0) / f_true + jitter_s
        peaks = np.where(polarity == 0, 1.1 + 0.2 * rng.random(n_cycles), 0.45 + 0.15 * rng.random(n_cycles))
        order = np.argsort(toa)
        return toa[order], peaks[order]

    if scenario == "missing_cycles":
        keep_mask = rng.random(n_cycles) > 0.22
        kept_cycles = cycles[keep_mask]
        phases_deg = rng.choice([42.0, 222.0], size=len(kept_cycles))
        jitter_s = rng.normal(0.0, 3.8e-6, size=len(kept_cycles))
        toa = kept_cycles / f_true + (phases_deg / 360.0) / f_true + jitter_s
        peaks = 0.6 + 0.8 * rng.random(len(kept_cycles))
        order = np.argsort(toa)
        return toa[order], peaks[order]

    raise ValueError(f"Unknown scenario: {scenario}")


def run_benchmark(
    *,
    output_dir: str | Path,
    n_trials: int = 40,
    seed: int = 42,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        "symmetric_moderate",
        "narrow_outliers",
        "asymmetric_polarity",
        "missing_cycles",
    ]
    methods = ("coherence", "harmonic_power", "epoch_folding", "gregory_loredo", "auto")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for scenario in scenarios:
        for trial_idx in range(n_trials):
            f_true = 50.0 + rng.uniform(-0.35, 0.35)
            n_cycles = int(rng.integers(1200, 2200))
            toa, peaks = _scenario_toa(scenario, rng=rng, f_true=f_true, n_cycles=n_cycles)

            start = time.perf_counter()
            results = compare_frequency_estimators(
                toa,
                base_freq=50.0,
                search_width=0.5,
                coarse_steps=2001,
                refine_half_width=0.02,
                max_events=12000,
                peak_weights=peaks,
                methods=methods,
                n_harmonics=4,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            for result in results:
                f_est = float(result["freq_hz"])
                rows.append(
                    {
                        "scenario": scenario,
                        "trial_idx": trial_idx,
                        "method": str(result["method"]),
                        "selected_method": str(result.get("selected_method", result["method"])),
                        "f_true_hz": f_true,
                        "f_est_hz": f_est,
                        "abs_error_hz": abs(f_est - f_true),
                        "score": float(result["score"]),
                        "sharpness": float(result.get("sharpness", float("nan"))),
                        "half_height_width_hz": float(result.get("half_height_width_hz", float("nan"))),
                        "common_axial_confidence": float(result.get("common_axial_confidence", float("nan"))),
                        "common_axial_peak_offset_hz": float(result.get("common_axial_peak_offset_hz", float("nan"))),
                        "n_events": int(len(toa)),
                        "elapsed_ms_shared": elapsed_ms,
                    }
                )

    detail_df = pd.DataFrame(rows)
    detail_csv = output_dir / "blind_prpd_benchmark_detail.csv"
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    summary_df = (
        detail_df.groupby(["scenario", "method"], as_index=False)
        .agg(
            trials=("trial_idx", "count"),
            mean_abs_error_hz=("abs_error_hz", "mean"),
            median_abs_error_hz=("abs_error_hz", "median"),
            p95_abs_error_hz=("abs_error_hz", lambda s: float(np.quantile(s, 0.95))),
            mean_score=("score", "mean"),
            mean_sharpness=("sharpness", "mean"),
            mean_half_height_width_hz=("half_height_width_hz", "mean"),
            mean_common_axial_confidence=("common_axial_confidence", "mean"),
            mean_common_axial_peak_offset_hz=("common_axial_peak_offset_hz", "mean"),
            mean_elapsed_ms_shared=("elapsed_ms_shared", "mean"),
        )
        .sort_values(["scenario", "mean_abs_error_hz", "method"])
        .reset_index(drop=True)
    )
    summary_csv = output_dir / "blind_prpd_benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    winner_df = (
        summary_df.sort_values(["scenario", "mean_abs_error_hz", "p95_abs_error_hz", "method"])
        .groupby("scenario", as_index=False)
        .first()
    )

    lines = [
        "# Blind PRPD benchmark",
        "",
        "## Scenarios",
        "",
        "- `symmetric_moderate`: two balanced lobes with moderate jitter.",
        "- `narrow_outliers`: narrow antipodal lobes plus low-amplitude outliers.",
        "- `asymmetric_polarity`: one polarity dominates in event count and amplitude.",
        "- `missing_cycles`: random cycle dropout with broader jitter.",
        "",
        "## Methods",
        "",
        "- `coherence`: weighted doubled-phase resultant.",
        "- `harmonic_power`: multi-harmonic doubled-phase concentration.",
        "- `epoch_folding`: folded-phase histogram concentration against uniform occupancy.",
        "- `gregory_loredo`: Bayesian event folding with piecewise-constant phase-rate bins.",
        "- `auto`: selector that compares `coherence`, `harmonic_power`, and `epoch_folding` by axial concentration.",
        "",
        "## Winners by scenario",
        "",
        "| Scenario | Winner | Mean abs error (Hz) | P95 abs error (Hz) |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, row in winner_df.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['mean_abs_error_hz']:.6f} | {row['p95_abs_error_hz']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Full summary",
            "",
            "| Scenario | Method | Mean abs error (Hz) | Median abs error (Hz) | P95 abs error (Hz) | Mean score | Mean common conf. | Mean peak offset (Hz) | Mean shared runtime (ms) |",
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
                    f"{row['mean_abs_error_hz']:.6f}",
                    f"{row['median_abs_error_hz']:.6f}",
                    f"{row['p95_abs_error_hz']:.6f}",
                    f"{row['mean_score']:.6f}",
                    f"{row['mean_common_axial_confidence']:.6f}",
                    f"{row['mean_common_axial_peak_offset_hz']:.6f}",
                    f"{row['mean_elapsed_ms_shared']:.2f}",
                ]
            )
            + " |"
        )

    summary_md = output_dir / "blind_prpd_benchmark_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark blind PRPD frequency estimators.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "outputs" / "blind_prpd_benchmark",
        help="Directory where benchmark outputs will be written.",
    )
    parser.add_argument("--trials", type=int, default=40, help="Number of trials per scenario.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    outputs = run_benchmark(output_dir=args.output_dir, n_trials=args.trials, seed=args.seed)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
