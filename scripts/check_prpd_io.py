from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deltapd.blind_prpd import reconstruct_blind_prpd
from deltapd.campaign.plot_material import plot_blind_prpd
from deltapd.descriptors import detect_pulses
from deltapd.loader import load_empirical_signal


@dataclass
class CheckResult:
    kind: str
    source: str
    fs_hz: float
    n_samples: int
    n_pulses: int
    n_phases: int
    phase_min_deg: float
    phase_max_deg: float
    peak_mean: float
    output_png: str
    status: str


def _run_case(
    kind: str,
    source: Path,
    out_dir: Path,
    threshold_sigma: float,
    min_separation_s: float,
) -> CheckResult:
    if kind == "csv":
        signal, fs, abs_times = load_empirical_signal(
            str(source),
            preserve_amplitude=True,
            include_absolute_times=True,
        )
        pulse_idx = detect_pulses(
            signal,
            fs,
            threshold_sigma=threshold_sigma,
            min_separation_s=min_separation_s,
            method="threshold",
        )
        toa_s = abs_times[pulse_idx] if abs_times is not None else pulse_idx / fs
    elif kind == "mat":
        signal, fs = load_empirical_signal(
            str(source),
            preserve_amplitude=True,
        )
        pulse_idx = detect_pulses(
            signal,
            fs,
            threshold_sigma=threshold_sigma,
            min_separation_s=min_separation_s,
            method="threshold",
        )
        toa_s = pulse_idx / fs
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    peaks = np.abs(signal[pulse_idx])
    out_png = out_dir / f"prpd_{kind}_check.png"

    if len(toa_s) < 10:
        return CheckResult(
            kind=kind.upper(),
            source=str(source),
            fs_hz=float(fs),
            n_samples=int(len(signal)),
            n_pulses=int(len(pulse_idx)),
            n_phases=0,
            phase_min_deg=float("nan"),
            phase_max_deg=float("nan"),
            peak_mean=float("nan"),
            output_png=str(out_png),
            status="FAIL: insufficient pulses for PRPD",
        )

    phases_deg, peaks_out = reconstruct_blind_prpd(
        toa_s,
        peaks,
        freq_hz=50.0,
        auto_calibrate=True,
    )
    df_plot = pd.DataFrame({"prpd_phase_deg": phases_deg, "peak_v": peaks_out})
    plot_blind_prpd(df_plot, str(out_png))

    return CheckResult(
        kind=kind.upper(),
        source=str(source),
        fs_hz=float(fs),
        n_samples=int(len(signal)),
        n_pulses=int(len(pulse_idx)),
        n_phases=int(len(phases_deg)),
        phase_min_deg=float(np.min(phases_deg)),
        phase_max_deg=float(np.max(phases_deg)),
        peak_mean=float(np.mean(peaks_out)),
        output_png=str(out_png),
        status="OK",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-check pseudo-PRPD for one CSV and one MAT input."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas/Prueba 1 - Internas/CH3.csv"),
        help="Input CSV waveform path.",
    )
    parser.add_argument(
        "--mat",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas/DeltaPD-main/SignalTestEnvolpe01.mat"),
        help="Input MAT waveform path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Output folder for PNGs and summary CSV.",
    )
    parser.add_argument(
        "--csv-threshold-sigma",
        type=float,
        default=5.0,
        help="Threshold sigma for CSV pulse detection.",
    )
    parser.add_argument(
        "--mat-threshold-sigma",
        type=float,
        default=5.0,
        help="Threshold sigma for MAT pulse detection.",
    )
    parser.add_argument(
        "--csv-min-sep-s",
        type=float,
        default=20e-9,
        help="Minimum pulse separation for CSV in seconds.",
    )
    parser.add_argument(
        "--mat-min-sep-s",
        type=float,
        default=1e-5,
        help="Minimum pulse separation for MAT in seconds.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _run_case(
            "csv",
            args.csv,
            args.out_dir,
            threshold_sigma=args.csv_threshold_sigma,
            min_separation_s=args.csv_min_sep_s,
        ),
        _run_case(
            "mat",
            args.mat,
            args.out_dir,
            threshold_sigma=args.mat_threshold_sigma,
            min_separation_s=args.mat_min_sep_s,
        ),
    ]

    for r in results:
        print(
            f"[{r.kind}] {r.status} | pulses={r.n_pulses} | phases={r.n_phases} | "
            f"fs={r.fs_hz:.6g} | png={r.output_png}"
        )

    summary = pd.DataFrame([asdict(r) for r in results])
    summary_path = args.out_dir / "prpd_io_check_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    return 0 if all(r.status == "OK" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
