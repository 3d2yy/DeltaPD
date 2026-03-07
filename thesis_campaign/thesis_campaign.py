from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from thesis_campaign.aggregate import summarize_metrics
from thesis_campaign.config import load_config
from thesis_campaign.detection_curves import compute_detection_curves, summarize_detection_curves
from thesis_campaign.metrics_spectral import compute_spectral_metrics
from thesis_campaign.metrics_time import compute_time_metrics
from deltapd.descriptors import extract_delta_t_vector
from deltapd.loader import load_empirical_signal
from deltapd.signal_model import wavelet_denoise_parametric


def _resolve_dataset_folder(base_dir: Path, folder: str) -> Path:
    dataset_dir = base_dir / folder
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    return dataset_dir


def _iter_channel_files(dataset_dir: Path, channel_name: str) -> list[Path]:
    exact = sorted(dataset_dir.rglob(f"{channel_name}.csv"))
    if exact:
        return exact
    generic = sorted(dataset_dir.rglob(f"*{channel_name}*.csv"))
    return generic


def _maybe_denoise(signal, denoise_cfg: dict[str, Any]):
    if not denoise_cfg.get("enabled", False):
        return signal
    return wavelet_denoise_parametric(
        signal,
        wavelet=denoise_cfg.get("wavelet", "db4"),
        threshold_mode=denoise_cfg.get("threshold_mode", "soft"),
        threshold_rule=denoise_cfg.get("threshold_rule", "universal"),
    )


def run_thesis_campaign(config_path: str | Path) -> dict[str, pd.DataFrame]:
    cfg = load_config(config_path)
    config_path = Path(config_path)
    repo_root = config_path.resolve().parents[2] if len(config_path.resolve().parents) >= 2 else config_path.resolve().parent

    base_dir = Path(cfg["base_dir"]).expanduser()
    output_dir = repo_root / cfg.get("output_dir", "outputs/thesis")
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = cfg.get("analysis_params", {})
    preserve_amplitude = bool(analysis.get("preserve_amplitude", True))
    default_fs = float(analysis.get("default_fs", 1e9))
    noise_window_ns = analysis.get("noise_window_ns")
    bands_ghz = analysis.get("bands_ghz", [[0.0, 0.5], [0.5, 1.5], [1.5, 2.4]])
    k_values = analysis.get("detection_k_values", [3.0, 5.0, 7.0])
    denoise_cfg = analysis.get("denoise", {"enabled": False})
    threshold_sigma = float(analysis.get("delta_t_threshold_sigma", 3.0))
    detection_method = str(analysis.get("delta_t_detection_method", "cfar"))

    rows: list[dict[str, Any]] = []

    for dataset_key, dataset_cfg in cfg.get("datasets", {}).items():
        dataset_dir = _resolve_dataset_folder(base_dir, dataset_cfg["folder"])
        dataset_label = dataset_cfg.get("label", dataset_cfg["folder"])
        mode = dataset_cfg.get("mode", "benchmark")
        channel_map = dataset_cfg.get("channel_map", {})

        for channel_name, antenna_name in channel_map.items():
            files = _iter_channel_files(dataset_dir, channel_name)
            for file_path in files:
                signal, fs, t_trig = load_empirical_signal(
                    str(file_path),
                    default_fs=default_fs,
                    preserve_amplitude=preserve_amplitude,
                    include_trigger_time=True,
                )
                signal_proc = _maybe_denoise(signal, denoise_cfg)

                time_metrics = compute_time_metrics(signal, fs, noise_window_ns=noise_window_ns)
                spectral_metrics = compute_spectral_metrics(signal_proc, fs, bands_ghz=bands_ghz)

                try:
                    delta_t = extract_delta_t_vector(
                        signal_proc,
                        fs,
                        threshold_sigma=threshold_sigma,
                        detection_method=detection_method,
                    )
                    delta_t_count = int(len(delta_t))
                    mean_delta_t_s = float(delta_t.mean()) if len(delta_t) else float("nan")
                except Exception:
                    delta_t_count = 0
                    mean_delta_t_s = float("nan")

                row = {
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_label,
                    "mode": mode,
                    "channel": channel_name,
                    "antenna": antenna_name,
                    "file_path": str(file_path),
                    "fs_hz": float(fs),
                    "trigger_time_epoch": float(t_trig),
                    "delta_t_count": delta_t_count,
                    "mean_delta_t_s": mean_delta_t_s,
                }
                row.update(time_metrics)
                row.update(spectral_metrics)
                rows.append(row)

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(metrics_df)
    curves_df = compute_detection_curves(metrics_df, k_values=k_values)
    curves_summary_df = summarize_detection_curves(curves_df)

    metrics_df.to_csv(output_dir / "thesis_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "thesis_summary_by_dataset_antenna.csv", index=False, encoding="utf-8-sig")
    curves_df.to_csv(output_dir / "thesis_detection_curves.csv", index=False, encoding="utf-8-sig")
    curves_summary_df.to_csv(output_dir / "thesis_detection_summary_3_5_7sigma.csv", index=False, encoding="utf-8-sig")

    return {
        "metrics": metrics_df,
        "summary": summary_df,
        "detection_curves": curves_df,
        "detection_summary": curves_summary_df,
    }
