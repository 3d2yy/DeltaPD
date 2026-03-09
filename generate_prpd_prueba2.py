from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from deltapd.loader import load_empirical_signal
from deltapd.descriptors import detect_pulses
from deltapd.blind_prpd import reconstruct_blind_prpd
from deltapd.campaign.plot_material import plot_blind_prpd


def build_prpd_prueba2(
    csv_path: Path,
    out_png: Path,
    threshold_sigma: float,
    min_separation_s: float,
) -> None:
    signal, fs_hz, times_abs_s = load_empirical_signal(
        str(csv_path),
        preserve_amplitude=True,
        include_absolute_times=True,
    )

    pulse_idx = detect_pulses(
        signal,
        fs_hz,
        threshold_sigma=threshold_sigma,
        min_separation_s=min_separation_s,
        method="threshold",
    )

    toa_s = times_abs_s[pulse_idx]
    peaks_v = np.abs(signal[pulse_idx])

    phases_deg, peaks_out = reconstruct_blind_prpd(
        toa_s,
        peaks_v,
        freq_hz=50.0,
        auto_calibrate=True,
    )

    df_plot = pd.DataFrame({"prpd_phase_deg": phases_deg, "peak_v": peaks_out})
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plot_blind_prpd(df_plot, str(out_png))

    print(f"CSV={csv_path}")
    print(f"FS_HZ={fs_hz}")
    print(f"N_SAMPLES={len(signal)}")
    print(f"N_PULSES={len(pulse_idx)}")
    print(f"N_PHASES={len(phases_deg)}")
    print(f"PHASE_MIN_DEG={float(np.min(phases_deg))}")
    print(f"PHASE_MAX_DEG={float(np.max(phases_deg))}")
    print(f"PEAK_MEAN={float(np.mean(peaks_out))}")
    print(f"OUT_PNG={out_png}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera PRPD real para Prueba 2 (CH3).")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas/Prueba 2 - Superficiales/CH3.csv"),
        help="Ruta del CSV de Prueba 2.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/prpd_p2_ch3.png"),
        help="Ruta de salida del PNG PRPD.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=5.0,
        help="Umbral en sigma para deteccion de pulsos.",
    )
    parser.add_argument(
        "--min-separation-s",
        type=float,
        default=20e-9,
        help="Separacion minima entre pulsos en segundos.",
    )
    args = parser.parse_args()

    build_prpd_prueba2(
        csv_path=args.csv,
        out_png=args.out,
        threshold_sigma=args.threshold_sigma,
        min_separation_s=args.min_separation_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
