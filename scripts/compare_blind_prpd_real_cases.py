from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from deltapd.blind_prpd import compare_frequency_estimators


DATASET_KEYS = ("P1", "P2", "P3", "G1", "G2", "G3")


def _repo_root() -> Path:
    return _REPO_ROOT


def _axial_entropy_score(toa_s: np.ndarray, peak_v: np.ndarray, freq_hz: float, n_bins: int = 24) -> float:
    phases_folded = np.mod(np.asarray(toa_s, dtype=np.float64) * float(freq_hz) * 360.0, 180.0)
    weights = np.sqrt(np.abs(np.asarray(peak_v, dtype=np.float64)))
    hist, _ = np.histogram(phases_folded, bins=n_bins, range=(0.0, 180.0), weights=weights)
    total = float(np.sum(hist))
    if total <= 0.0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs))) / np.log(n_bins)
    return float(1.0 - entropy)


def _coherence_score(toa_s: np.ndarray, peak_v: np.ndarray, freq_hz: float) -> float:
    weights = np.sqrt(np.abs(np.asarray(peak_v, dtype=np.float64)))
    denom = max(float(np.sum(weights)), 1e-30)
    phasor = np.exp(1j * 4.0 * np.pi * float(freq_hz) * np.asarray(toa_s, dtype=np.float64))
    return float(np.abs(np.sum(weights * phasor)) / denom)


def run_real_case_comparison(
    *,
    state_alarm_root: str | Path,
    output_dir: str | Path,
    bootstrap_iterations: int = 6,
    bootstrap_sample_fraction: float = 0.75,
    bootstrap_seed: int = 42,
) -> dict[str, Path]:
    state_alarm_root = Path(state_alarm_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []
    for dataset_key in DATASET_KEYS:
        csv_path = state_alarm_root / dataset_key / "material" / "delta_t_series_master.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["toa_s", "peak_v"])
        toa = df["toa_s"].to_numpy(dtype=np.float64)
        peak_v = df["peak_v"].to_numpy(dtype=np.float64)
        results = compare_frequency_estimators(
            toa,
            peak_weights=peak_v,
        methods=("coherence", "harmonic_power", "epoch_folding", "h_test", "pdm", "gregory_loredo", "auto"),
            n_harmonics=4,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_sample_fraction=bootstrap_sample_fraction,
            bootstrap_seed=bootstrap_seed,
        )
        for result in results:
            freq_hz = float(result["freq_hz"])
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "method": str(result["method"]),
                    "selected_method": str(result.get("selected_method", result["method"])),
                    "freq_hz": freq_hz,
                    "method_score": float(result["score"]),
                    "sharpness": float(result.get("sharpness", float("nan"))),
                    "half_height_width_hz": float(result.get("half_height_width_hz", float("nan"))),
                    "common_axial_confidence": float(result.get("common_axial_confidence", float("nan"))),
                    "common_axial_peak_offset_hz": float(result.get("common_axial_peak_offset_hz", float("nan"))),
                    "bootstrap_freq_std_hz": float(result.get("bootstrap_freq_std_hz", float("nan"))),
                    "bootstrap_method_agreement": float(result.get("bootstrap_method_agreement", float("nan"))),
                    "winner_margin": float(result.get("winner_margin", float("nan"))),
                    "axial_entropy_score": _axial_entropy_score(toa, peak_v, freq_hz),
                    "coherence_common_score": _coherence_score(toa, peak_v, freq_hz),
                    "n_events": int(len(toa)),
                }
            )

    detail_df = pd.DataFrame(rows)
    detail_csv = output_dir / "blind_prpd_real_case_comparison.csv"
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# Blind PRPD real-case comparison",
        "",
        "| Dataset | Method | Winner | Freq (Hz) | Method score | Common conf. | Peak offset (Hz) | Boot std (Hz) | Boot agree. | Axial entropy score | Common coherence | Events |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in detail_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset_key"]),
                    str(row["method"]),
                    str(row["selected_method"]),
                    f"{row['freq_hz']:.6f}",
                    f"{row['method_score']:.6f}",
                    f"{row['common_axial_confidence']:.6f}",
                    f"{row['common_axial_peak_offset_hz']:.6f}",
                    f"{row['bootstrap_freq_std_hz']:.6f}",
                    f"{row['bootstrap_method_agreement']:.6f}",
                    f"{row['axial_entropy_score']:.6f}",
                    f"{row['coherence_common_score']:.6f}",
                    str(int(row["n_events"])),
                ]
            )
            + " |"
        )

    summary_path = output_dir / "blind_prpd_real_case_comparison.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "detail_csv": detail_csv,
        "summary_md": summary_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare blind PRPD methods on real state/alarm cases.")
    parser.add_argument(
        "--state-alarm-root",
        type=Path,
        default=_repo_root() / "outputs" / "state_alarm_ch3",
        help="Root folder containing per-case material outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "outputs" / "blind_prpd_real_cases",
        help="Directory where comparison tables will be written.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=6, help="Bootstrap iterations for blind PRPD stability.")
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=0.75, help="Bootstrap sample fraction.")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Bootstrap random seed.")
    args = parser.parse_args()

    outputs = run_real_case_comparison(
        state_alarm_root=args.state_alarm_root,
        output_dir=args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_sample_fraction=args.bootstrap_sample_fraction,
        bootstrap_seed=args.bootstrap_seed,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
