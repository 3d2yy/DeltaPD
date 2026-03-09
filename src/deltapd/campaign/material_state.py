from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import json
from datetime import datetime

from deltapd.descriptors import detect_pulses

from deltapd.statistics import (
    fit_weibull_moving, 
    compute_burstiness_index,
    compute_fano_factor,
    compute_phase_entropy
)
from deltapd.blind_prpd import compute_local_blind_prpd_trace, reconstruct_blind_prpd
from deltapd.classifier import classify_events
from deltapd.campaign.event_pipeline import load_and_extract_event_series

from deltapd.campaign.metrics_time import compute_time_metrics
from deltapd.campaign.metrics_spectral import compute_spectral_metrics
from deltapd.campaign.plot_material import (
    plot_raw_with_detections,
    plot_delta_t_series,
    plot_delta_t_histogram,
    plot_rate_series,
    plot_rolling_stats,
    plot_ewma_cusum,
)
from deltapd.campaign.plot_material import (
    plot_advanced_analytics,
    plot_blind_prpd
)

def export_sensitivity_report(x: np.ndarray, fs: float, out_dir: Path, cfg: dict):
    from deltapd.descriptors import detect_pulses
    import pandas as pd
    
    print("[MATERIAL STATE] Generando reporte de sensibilidad...")
    k_sigmas = [4.0, 5.0, 6.0]
    max_dts = [0.1, 0.5, 1.0]
    
    min_sep = cfg["detection"]["refractory_ns"] * 1e-9
    
    results = []
    for k in k_sigmas:
        pulse_indices = detect_pulses(
            signal_data=x,
            fs=fs,
            threshold_sigma=k,
            min_separation_s=min_sep,
            method="threshold"
        )
        if len(pulse_indices) < 2:
            continue
            
        toa = pulse_indices / fs
        delta_t = np.diff(toa)
        
        for mdt in max_dts:
            outliers = delta_t > mdt
            valid = sum(~outliers)
            results.append({
                "k_sigma": k,
                "max_valid_dt_s": mdt,
                "total_events": len(delta_t),
                "valid_events": valid,
                "outliers": sum(outliers)
            })
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_dir / "sensitivity_report.csv", index=False)
        print(f" -> Reporte de sensibilidad exportado a {out_dir / 'sensitivity_report.csv'}")

def assign_stage_by_time(df: pd.DataFrame, boundaries_s: list[float]) -> pd.DataFrame:
    """
    boundaries_s = [t1, t2, t3, ...]
    """
    labels = []
    for t in df["toa_s"]:
        stage = 1
        for b in boundaries_s:
            if t > b:
                stage += 1
            else:
                break
        labels.append(stage)
    df["stage"] = labels
    return df


def _resolve_dataset_file(cfg: dict) -> Path:
    dataset_cfg = cfg["dataset"]
    direct_file = dataset_cfg.get("file_path")
    base_dir = Path(cfg.get("base_dir", ".")).expanduser()

    if direct_file:
        file_path = Path(direct_file).expanduser()
        if not file_path.is_absolute():
            file_path = base_dir / file_path
        return file_path

    folder = dataset_cfg["folder"]
    channel = dataset_cfg["channel"]
    return base_dir / folder / f"{channel}.csv"

def run_material_state(config_path: str | Path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolving workspace root
    config_path_obj = Path(config_path)
    repo_root = config_path_obj.resolve().parents[1] if len(config_path_obj.resolve().parents) >= 1 else config_path_obj.resolve().parent

    out_dir = repo_root / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = _resolve_dataset_file(cfg)
    print(f"\\n[MATERIAL STATE] Leyendo {file_path}...")

    if not file_path.exists():
        print(f"Error: No se encontrÃ³ el archivo de datos en {file_path}")
        return

    # 1. Cargar, preprocesar y detectar con el pipeline compartido
    min_sep = cfg["detection"]["refractory_ns"] * 1e-9
    extracted = load_and_extract_event_series(
        str(file_path),
        preserve_amplitude=cfg["preprocess"]["preserve_amplitude"],
        include_absolute_times=True,
        threshold_sigma=cfg["detection"]["k_sigma"],
        min_separation_s=min_sep,
        detection_method="threshold",
        wavelet_denoise=cfg["preprocess"].get("wavelet_denoise", False),
        is_envelope=cfg["preprocess"].get("is_envelope", False),
        wavelet="db4",
    )
    x = extracted.raw_signal
    signal_proc = extracted.processed_signal
    fs = extracted.fs_hz
    pulse_indices = extracted.pulse_indices
    t = np.arange(len(x)) / fs

    if len(extracted.event_delta_t_s) == 0:
        print("[MATERIAL STATE] No hay suficientes pulsos detectados para anÃ¡lisis Î”t.")
        return

    # Reporte de sensibilidad opcional
    if cfg["analysis"].get("export_sensitivity_report", False):
        export_sensitivity_report(signal_proc, fs, out_dir, cfg)

    toa = extracted.event_toa_s
    delta_t = extracted.event_delta_t_s
    peaks = extracted.event_peaks_v
    
    # 3. AnalÃ­tica Avanzada Q1 (PRPD, Weibull, RÃ¡fagas)
    blind_cfg = cfg["analysis"].get("blind_prpd", {})
    blind_method = str(blind_cfg.get("calibration_method", "auto"))
    blind_harmonics = int(blind_cfg.get("n_harmonics", 4))
    blind_search_width = float(blind_cfg.get("search_width_hz", 0.5))
    blind_coarse_steps = int(blind_cfg.get("coarse_steps", 2001))
    blind_refine_half_width = float(blind_cfg.get("refine_half_width_hz", 0.02))
    blind_max_events = int(blind_cfg.get("max_events", 20000))
    blind_robust_refine = bool(blind_cfg.get("robust_refine", True))
    blind_bootstrap_iterations = int(blind_cfg.get("bootstrap_iterations", 0))
    blind_bootstrap_fraction = float(blind_cfg.get("bootstrap_sample_fraction", 0.75))
    blind_bootstrap_seed = blind_cfg.get("bootstrap_seed")
    blind_local_window_size = int(blind_cfg.get("local_window_size_events", 0))
    blind_local_window_step = int(blind_cfg.get("local_window_step_events", 0))
    blind_local_min_events = int(blind_cfg.get("local_min_events_per_window", 128))
    blind_local_min_windows = int(blind_cfg.get("local_min_window_count", 3))
    blind_display_center = blind_cfg.get("display_center_deg")
    phases_deg, _, blind_info = reconstruct_blind_prpd(
        toa,
        peaks,
        freq_hz=50.0,
        auto_calibrate=True,
        calibration_method=blind_method,
        calibration_search_width_hz=blind_search_width,
        calibration_coarse_steps=blind_coarse_steps,
        calibration_refine_half_width_hz=blind_refine_half_width,
        calibration_max_events=blind_max_events,
        calibration_robust_refine=blind_robust_refine,
        n_harmonics=blind_harmonics,
        bootstrap_iterations=blind_bootstrap_iterations,
        bootstrap_sample_fraction=blind_bootstrap_fraction,
        bootstrap_seed=None if blind_bootstrap_seed is None else int(blind_bootstrap_seed),
        local_window_size_events=blind_local_window_size,
        local_window_step_events=blind_local_window_step,
        local_min_events_per_window=blind_local_min_events,
        local_min_window_count=blind_local_min_windows,
        display_center_deg=blind_display_center,
        return_metadata=True,
    )
    blind_local_trace = compute_local_blind_prpd_trace(
        toa,
        base_freq=50.0,
        search_width=blind_search_width,
        coarse_steps=blind_coarse_steps,
        refine_half_width=blind_refine_half_width,
        max_events=blind_max_events,
        peak_weights=peaks,
        robust_refine=blind_robust_refine,
        method=blind_method,
        n_harmonics=blind_harmonics,
        window_size_events=blind_local_window_size,
        step_events=blind_local_window_step,
        min_events_per_window=blind_local_min_events,
        min_window_count=blind_local_min_windows,
        global_freq_hz=blind_info.calibrated_freq_hz,
    )
    if blind_local_trace:
        pd.DataFrame(blind_local_trace).to_csv(
            out_dir / "blind_prpd_local_trace.csv",
            index=False,
        )

    print(f" -> ExtraÃ­dos {len(delta_t)} pulsos (delta t)")

    # 4. Serie de Î”t y TOA
    # El usuario pide columnas por evento/ventana, el DF base contendrÃ¡ a los eventos:
    df_delta = pd.DataFrame({
        "event_idx": np.arange(1, len(delta_t) + 1),
        "toa_s": toa,
        "delta_t_s": delta_t,
        "log10_dt": np.log10(np.maximum(delta_t, 1e-12)),
        "pulse_rate_hz": 1.0 / np.maximum(delta_t, 1e-12),
        "peak_v": peaks,
        "prpd_phase_deg": phases_deg
    })

    # Filtrar macro-outliers para estadÃ­sticas mÃ³viles
    max_valid_dt = cfg["analysis"].get("max_valid_dt_s", 1.0)
    df_delta["is_outlier"] = df_delta["delta_t_s"] > max_valid_dt
    
    # Calcular estadÃ­sticas mÃ³viles segÃºn cfg analysis SOLO sobre datos vÃ¡lidos (evita que outliers aplasten la curva)
    w = cfg["analysis"]["rolling_window_events"]
    valid_dt = df_delta["delta_t_s"].where(~df_delta["is_outlier"])
    
    df_delta["rolling_rate_hz"] = (1.0 / valid_dt).rolling(w, min_periods=max(5, w//5)).mean()
    df_delta["rolling_median_dt"] = valid_dt.rolling(w, min_periods=max(5, w//5)).median()
    
    q3 = valid_dt.rolling(w, min_periods=max(5, w//5)).quantile(0.75)
    q1 = valid_dt.rolling(w, min_periods=max(5, w//5)).quantile(0.25)
    df_delta["rolling_iqr_dt"] = q3 - q1

    df_delta["rolling_p90_dt"] = valid_dt.rolling(w, min_periods=max(5, w//5)).quantile(0.90)
    
    std = valid_dt.rolling(w, min_periods=max(5, w//5)).std()
    mean = valid_dt.rolling(w, min_periods=max(5, w//5)).mean()
    df_delta["rolling_cv_dt"] = std / mean
    
    # Analítica Avanzada Dinámica
    beta, eta = fit_weibull_moving(valid_dt.to_numpy(), window=w, min_periods=max(5, w//5))
    burst_idx = compute_burstiness_index(valid_dt.to_numpy(), window=w, min_periods=max(5, w//5))
    
    # 3.2 Nuevos indicadores: Fano y Entropía
    _, fano_vals = compute_fano_factor(toa, bin_duration_s=0.1, window_bins=20, min_bins=5)
    # Alinear fano_vals (que es por bin) con df_delta (que es por evento)
    # Para simplicidad en el DF maestro, usamos la entropía que ya es por evento
    entropy_vals = compute_phase_entropy(phases_deg, window=w, min_periods=max(5, w//5))
    
    df_delta["rolling_weibull_beta"] = beta
    df_delta["rolling_burstiness"] = burst_idx
    df_delta["rolling_entropy"] = entropy_vals
    
    # 3.3 Clasificación Automática
    print(" -> Ejecutando clasificador de 5 indicadores...")
    indicator_dict = {
        'weibull_beta': beta,
        'burstiness': burst_idx,
        'cv': df_delta["rolling_cv_dt"].to_numpy(),
        'entropy': entropy_vals
        # Fano se añade si se desea por evento, por ahora 4/5 para el clasificador por evento
    }
    categories, confidences = classify_events(indicator_dict)
    df_delta["pred_category"] = categories
    df_delta["pred_confidence"] = confidences

    # Asignar etapa si estÃ¡n en el config y stage_aware es True
    stage_aware = cfg["analysis"].get("stage_aware", False)
    df_delta["window_id"] = np.arange(len(df_delta)) // w

    if stage_aware and "stage_boundaries_s" in cfg["analysis"]:
        b_cfg = cfg["analysis"]["stage_boundaries_s"]
        if b_cfg == "auto":
            # Dividir la seÃ±al real en 3 fragmentos equitativos temporales basados en el rango activo
            min_t = df_delta["toa_s"].min()
            max_t = df_delta["toa_s"].max()
            span = max_t - min_t
            boundaries = [min_t + span * 0.33, min_t + span * 0.66]
            group_col = "segment"
            # Borrar residual de etapas previas para no confundir al usuario
            if (out_dir / "stage_summary.csv").exists():
                (out_dir / "stage_summary.csv").unlink()
        else:
            boundaries = b_cfg
            group_col = "stage"
            # Borrar residual de segmentos paramÃ©tricos previos
            if (out_dir / "segment_summary.csv").exists():
                (out_dir / "segment_summary.csv").unlink()
            
        df_delta = assign_stage_by_time(df_delta, boundaries)
        if group_col == "segment":
            df_delta.rename(columns={"stage": "segment"}, inplace=True)

    def _export_summary(group_col: str):
        if group_col not in df_delta.columns:
            return
        print(f" -> Calculando resÃºmenes de energÃ­a agrupando por {group_col}...")
        summary_rows = []
        for g_id, df_g in df_delta.groupby(group_col):
            if len(df_g) == 0: continue
            
            t_start = df_g["toa_s"].min()
            t_end = df_g["toa_s"].max()
            
            sig_chunk = signal_proc[int(max(0, t_start*fs)): int(min(len(signal_proc), t_end*fs))]
            metrics = {}
            if len(sig_chunk) > 100:
                metrics = compute_time_metrics(sig_chunk, fs, cfg["detection"]["noise_window_ns"])
                
            vd_g = df_g["delta_t_s"].where(~df_g["is_outlier"]).dropna()
            n_valid = len(df_g) - int(df_g["is_outlier"].sum())
            
            if n_valid == 0:
                print(f" [WARNING] El grupo '{g_id}' ({group_col}) tiene 0 eventos vÃ¡lidos. No interpretar como rÃ©gimen comparable.")
                
            row = {
                group_col: g_id,
                "n_events": len(df_g),
                "n_valid_events": n_valid,
                "n_outliers": df_g["is_outlier"].sum(),
                "median_dt": vd_g.median() if len(vd_g) else 0.0,
                "iqr_dt": (vd_g.quantile(0.75) - vd_g.quantile(0.25)) if len(vd_g) else 0.0,
                "p90_dt": vd_g.quantile(0.90) if len(vd_g) else 0.0,
                "cv_dt": vd_g.std() / vd_g.mean() if len(vd_g) and vd_g.mean() > 0 else 0,
                "mean_smoothed_rate": df_g["rolling_rate_hz"].mean(),
                "mean_energy": metrics.get("energy", 0.0),
                "mean_snr_db": metrics.get("snr_db", 0.0)
            }
            summary_rows.append(row)
            
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(out_dir / f"{group_col}_summary.csv", index=False)

    _export_summary("window_id")
    if stage_aware:
        _export_summary(group_col)

    # Exportar master dataset
    df_delta.to_csv(out_dir / "delta_t_series_master.csv", index=False)

    print(f" -> Datos tabulares guardados en {out_dir}")

    # 5. Exportar grÃ¡ficas
    if cfg["plots"]["show_raw_with_detections"]:
        # Para evitar colgar matplotlib, graficamos solo una porciÃ³n representativa
        limit = min(int(1e6), len(x))
        valid_toa = toa[toa < (limit / fs)]
        plot_raw_with_detections(t[:limit], x[:limit], valid_toa, str(out_dir / "01_raw_with_detections.png"))

    if cfg["plots"]["show_delta_t_series"]:
        plot_delta_t_series(df_delta, str(out_dir / "02a_delta_t_series_lineal.png"), is_log=False)
        plot_delta_t_series(df_delta, str(out_dir / "02b_delta_t_series_log10.png"), is_log=True)

    if cfg["plots"]["show_delta_t_hist"]:
        plot_delta_t_histogram(df_delta, str(out_dir / "03a_delta_t_histogram_lineal.png"), is_log=False)
        plot_delta_t_histogram(df_delta, str(out_dir / "03b_delta_t_histogram_log10.png"), is_log=True)

    if cfg["plots"]["show_rate_series"]:
        plot_rate_series(df_delta, str(out_dir / "04_event_rate_series.png"))

    if cfg["plots"]["show_rolling_stats"]:
        plot_rolling_stats(df_delta, str(out_dir / "05_rolling_delta_t_stats.png"))

    if cfg["plots"]["show_ewma_cusum"]:
        plot_ewma_cusum(
            df_delta,
            alpha=cfg["analysis"]["ewma_alpha"],
            cusum_k=cfg["analysis"]["cusum_k"],
            cusum_h=cfg["analysis"]["cusum_h"],
            out_png=str(out_dir / "06_ewma_cusum_robusto.png"),
        )
        
    if cfg["plots"].get("show_advanced_stats", False):
        try:
            plot_advanced_analytics(df_delta, str(out_dir / "07_advanced_q1_stats.png"))
        except NameError:
            pass # fallback if not imported
            
    if cfg["plots"].get("show_blind_prpd", False):
        try:
            plot_blind_prpd(df_delta, str(out_dir / "08_blind_prpd_50hz.png"))
        except NameError:
            pass
            
    if "pred_category" in df_delta.columns:
        from deltapd.campaign.plot_material import plot_classification_timeline
        plot_classification_timeline(df_delta, str(out_dir / "09_classification_trend.png"))

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config_used": cfg,
        "source_file": str(file_path),
        "total_events": len(df_delta),
        "valid_events": int((~df_delta["is_outlier"]).sum()),
        "outlier_events": int(df_delta["is_outlier"].sum()),
        "blind_prpd": blind_info.to_dict(),
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    print(f" -> Manifest de corrida exportado a {out_dir / 'run_manifest.json'}")
    
    print(f" -> GrÃ¡ficas Material State exportadas exitosamente.\\n[MATERIAL STATE] Finalizado.")

