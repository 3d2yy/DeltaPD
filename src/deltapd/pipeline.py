"""
Main entry point — UHF-PD validation pipeline.

Executes the four-phase numerical workflow:
    Phase 1: Stochastic wavelet optimisation (Monte Carlo + grid search)
    Phase 2: Variable isolation via inter-pulse interval extraction (Δt)
    Phase 3: Tracking with Kalman, adaptive EWMA, and CUSUM
    Phase 4: Quantification (empirical Big-O + convergence/FPR confusion matrix)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from deltapd.decision_layer import campaign_summary, evaluate_campaign
from deltapd.descriptors import extract_delta_t_vector
from deltapd.signal_model import (
    generate_uhf_pd_signal_physical,
    monte_carlo_wavelet_optimization,
    wavelet_denoise_parametric,
)
from deltapd.trackers import apply_delta_t_tracking
from deltapd.validation import (
    generate_convergence_confusion_matrix,
    generate_phase4_report,
    measure_all_tracking_complexities,
)
from deltapd.loader import load_empirical_signal

# ===================================================================
# Pipeline helpers
# ===================================================================


def run_phase1(
    n_samples: int = 5000000,
    fs: float = 5e9,
    n_iterations: int = 500,
    seed: int = 42,
    verbose: bool = True,
    empirical_path: Optional[str] = None,
):
    """Phase 1 — Stochastic wavelet optimisation (empirical or synthetic).

    Returns
    -------
    mc_result : MonteCarloResult
        Optimal wavelet configuration and full grid.
    clean : ndarray
        Clean reference signal used for optimisation.
    noisy : ndarray
        Noisy copy used for optimisation.
    """
    if verbose:
        print("=" * 70)
        print("PHASE 1 — Stochastic Wavelet Optimisation")
        print("=" * 70)

    if empirical_path is not None:
        if verbose:
            print(f"  Loading empirical signal from: {empirical_path}")
        try:
            noisy, loaded_fs = load_empirical_signal(empirical_path, default_fs=fs)
            fs = loaded_fs  # Override fs with the one from the file if available
            clean = noisy   # For empirical, we assume the input is the reference
            if len(noisy) > n_samples:
                 noisy = noisy[:n_samples]
                 clean = clean[:n_samples]
            n_samples = len(noisy)
        except Exception as e:
            if verbose:
                print(f"  Failed to load empirical signal: {e}")
                print("  Falling back to synthetic generation...")
            clean, noisy = generate_uhf_pd_signal_physical(
                n_samples=n_samples, fs=fs, seed=seed,
            )
    else:
        clean, noisy = generate_uhf_pd_signal_physical(
            n_samples=n_samples,
            fs=fs,
            seed=seed,
        )

    mc_result = monte_carlo_wavelet_optimization(
        reference_clean=clean,
        fs=fs,
        n_iterations=n_iterations,
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        print(
            f"\n  Optimal config: wavelet={mc_result.best_wavelet}, "
            f"mode={mc_result.best_threshold_mode}, "
            f"rule={mc_result.best_threshold_rule}"
        )
        print(
            f"  E[RMSE]={mc_result.best_rmse_mean:.6f}, "
            f"Var[RMSE]={mc_result.best_rmse_var:.2e}, "
            f"converged={mc_result.converged}"
        )

    return mc_result, clean, noisy


def run_phase2(
    noisy_signal,
    fs: float = 5e9,
    mc_result=None,
    threshold_sigma: float = 3.0,
    detection_method: str = "cfar",
    verbose: bool = True,
):
    """Phase 2 — delta-t vector extraction (variable isolation).

    Parameters
    ----------
    noisy_signal : ndarray
        Signal to process.
    fs : float
        Sampling frequency.
    mc_result : MonteCarloResult, optional
        If provided, applies the optimal wavelet denoising before extraction.
    threshold_sigma : float
        Pulse detection threshold in multiples of σ.

    Returns
    -------
    delta_t : ndarray
        1-D inter-pulse interval vector [Δt₁, Δt₂, …] in seconds.
    denoised : ndarray
        Denoised signal (if mc_result provided) or original.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2 -- delta-t Vector Extraction (Variable Isolation)")
        print("=" * 70)

    # Apply optimal wavelet denoising from Phase 1
    if mc_result is not None:
        denoised = wavelet_denoise_parametric(
            noisy_signal,
            wavelet=mc_result.best_wavelet,
            threshold_mode=mc_result.best_threshold_mode,
            threshold_rule=mc_result.best_threshold_rule,
        )
        if verbose:
            print(
                f"  Denoised with {mc_result.best_wavelet} / "
                f"{mc_result.best_threshold_mode} / "
                f"{mc_result.best_threshold_rule}"
            )
    else:
        denoised = noisy_signal

    delta_t = extract_delta_t_vector(
        denoised,
        fs,
        threshold_sigma=threshold_sigma,
        detection_method=detection_method,
    )

    if verbose:
        print(f"  Extracted delta-t vector: {len(delta_t)} intervals")
        if len(delta_t) > 0:
            print(
                f"  delta-t statistics: mean={np.mean(delta_t):.4e} s, "
                f"std={np.std(delta_t):.4e} s, "
                f"min={np.min(delta_t):.4e} s, "
                f"max={np.max(delta_t):.4e} s"
            )

    return delta_t, denoised


def run_phase3(delta_t, verbose: bool = True):
    """Phase 3 — Tracking evaluation (Kalman, EWMA, CUSUM).

    Returns
    -------
    tracking_result : DeltaTTrackingResult
        Aggregated tracking results from all three algorithms.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3 -- delta-t Tracking (Kalman / EWMA / CUSUM)")
        print("=" * 70)

    tracking_result = apply_delta_t_tracking(delta_t)

    if verbose:
        print(
            f"  Kalman: steady-state gain = "
            f"{tracking_result.kalman.steady_state_gain:.6f}"
        )
        print(f"  EWMA: final alpha = {tracking_result.ewma.alpha_sequence[-1]:.4f}")
        print(
            f"  CUSUM: {tracking_result.cusum.n_alarms} alarms "
            f"(threshold={tracking_result.cusum.threshold:.2f})"
        )

    return tracking_result


def run_phase4(
    sizes=(256, 512, 1024, 2048, 4096, 8192),
    n_repeats: int = 5,
    seed: int = 42,
    verbose: bool = True,
):
    """Phase 4 — Quantification (Big-O + convergence/FPR).

    Returns
    -------
    complexity : dict
        Per-algorithm Big-O estimates.
    confusion : ConvergenceConfusionMatrix
        Convergence-latency vs FPR matrix.
    report : str
        Human-readable Phase 4 report.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4 — Asymptotic Quantification")
        print("=" * 70)

    if verbose:
        print("  Measuring empirical Big-O complexities …")
    complexity = measure_all_tracking_complexities(
        sizes=sizes,
        n_repeats=n_repeats,
        seed=seed,
    )

    for name, est in complexity.items():
        if verbose:
            print(
                f"    {name}: O(n^{est.exponent_b:.2f})  " f"[R┬▓={est.r_squared:.4f}]"
            )

    if verbose:
        print("  Building convergence/FPR confusion matrix …")
    confusion = generate_convergence_confusion_matrix(seed=seed)

    report = generate_phase4_report(complexity, confusion)

    if verbose:
        print("\n" + report)

    return complexity, confusion, report


def main(
    n_samples: int = 4096,
    fs: float = 1e9,
    mc_iterations: int = 100,
    seed: int = 42,
    verbose: bool = True,
):
    """Legacy CLI entrypoint for the four-phase validation pipeline."""
    mc_result, _clean, noisy = run_phase1(
        n_samples=n_samples,
        fs=fs,
        n_iterations=mc_iterations,
        seed=seed,
        verbose=verbose,
    )
    delta_t, _denoised = run_phase2(noisy, fs=fs, mc_result=mc_result, verbose=verbose)
    tracking = run_phase3(delta_t, verbose=verbose)
    complexity, confusion, report = run_phase4(seed=seed, verbose=verbose)
    return {
        "mc_result": mc_result,
        "delta_t": delta_t,
        "tracking": tracking,
        "complexity": complexity,
        "confusion": confusion,
        "report": report,
    }


from pathlib import Path


def run_empirical_pipeline(
    campaign_dir: Path,
    target_filename: str = "*",
    fs: float = 5e9,
    is_envelope: bool = False,
):
    """Orchestrator for physical validation over temporal segments.

    Compensates inter-segment dead-time using absolute timestamps
    when available (e.g. from segmented oscilloscope captures).

    Parameters
    ----------
    campaign_dir : Path
        Directory containing waveform files.
    target_filename : str
        Glob pattern for file selection. Use '*' for all supported files.
    fs : float
        Default sampling frequency in Hz.
    is_envelope : bool
        If True, skip wavelet denoising (signal is already an envelope).

    Returns
    -------
    tuple or None
        (delta_t_final, kalman_result, cusum_result) or None if insufficient data.
    """
    print(f"\n[PIPELINE] Starting empirical campaign in: {campaign_dir.name}")

    valid_extensions = {".csv", ".mat", ".h5", ".hdf5"}

    if target_filename != "*":
        csv_files = list(campaign_dir.rglob(target_filename))
    else:
        csv_files = [f for f in campaign_dir.rglob("*")
                     if f.suffix.lower() in valid_extensions]

    if not csv_files:
        print(f"Error: No compatible files found in {campaign_dir}")
        return None

    from deltapd.descriptors import detect_pulses_cfar, compute_delta_t, extract_pulse_morphology
    from deltapd.signal_model import wavelet_denoise
    import pandas as pd
    from datetime import datetime

    segments = []
    print(f"[METADATA] Ingesting {len(csv_files)} segments...")
    for f_path in csv_files:
        voltage_array, estimated_fs, t_trig = load_empirical_signal(
            str(f_path), default_fs=fs, include_trigger_time=True,
        )
        if len(voltage_array) > 0:
            segments.append((t_trig, f_path, voltage_array))

    segments.sort(key=lambda x: x[0])
    global_delta_t = []
    global_morphology_dfs = []

    prev_t_trig = None
    prev_t_end = None

    for i, (t_trig, f_path, voltage_array) in enumerate(segments):
        print(f"\n[SEGMENT {i+1}/{len(segments)}] {f_path.name} (T_trig={t_trig:.2f})")

        if not is_envelope:
            denoised_signal = wavelet_denoise(voltage_array)
        else:
            denoised_signal = voltage_array

        pulse_indices = detect_pulses_cfar(denoised_signal, fs=fs)

        if len(pulse_indices) < 2:
            print(f"  Fewer than 2 pulses detected. Skipping segment.")
            continue

        t_start = pulse_indices[0] / fs
        t_end = pulse_indices[-1] / fs
        local_delta_t = compute_delta_t(pulse_indices, fs)

        df_morph = extract_pulse_morphology(denoised_signal, pulse_indices, fs)

        if not df_morph.empty:
            df_morph["segment_id"] = i + 1
            df_morph["file_name"] = f_path.name
            df_morph["t_trig_epoch"] = t_trig
            global_morphology_dfs.append(df_morph)

        # Dead-time compensation between segments
        if prev_t_trig is not None and prev_t_end is not None and t_trig > 0:
            dt_boundary = (t_trig + t_start) - (prev_t_trig + prev_t_end)
            if dt_boundary > 0:
                print(f"  Dead-time compensated: dt_boundary = {dt_boundary:.6f} s")
                global_delta_t.append(dt_boundary)

        global_delta_t.extend(local_delta_t.tolist())
        print(f"  Extracted {len(local_delta_t)} inter-pulse intervals.")

        prev_t_trig = t_trig
        prev_t_end = t_end

    delta_t_final = np.array(global_delta_t, dtype=np.float64)
    n_events = len(delta_t_final)

    if n_events < 5:
        print(f"\nInsufficient events ({n_events}) for tracking analysis.")
        return None

    print(f"\n[TRACKING] Running Kalman/EWMA/CUSUM on {n_events} intervals...")
    tracking_result = apply_delta_t_tracking(
        delta_t_final, fs=fs, cusum_drift=0.5, cusum_threshold=8.0,
    )

    # Diagnostic statistics (Z-score validation)
    z = tracking_result.kalman.z_scores
    z_mean = float(np.mean(z))
    z_std = float(np.std(z))
    pct_3sigma = float(np.mean(np.abs(z) > 3) * 100)
    n_alarms = tracking_result.cusum.n_alarms

    print(f"\n[Z-SCORE DIAGNOSTICS]")
    print(f"  mean(z)  = {z_mean:+.4f}  (ideal: 0.0)")
    print(f"  std(z)   = {z_std:.4f}   (ideal: 1.0)")
    print(f"  %|z|>3   = {pct_3sigma:.2f}%  (Gaussian expected: 0.27%)")
    print(f"  n_alarms = {n_alarms}")

    # Export results
    import pandas as pd
    from datetime import datetime

    project_root = Path(__file__).resolve().parent.parent.parent
    export_dir = project_root / "exports"
    export_dir.mkdir(exist_ok=True)

    df_results = pd.DataFrame({
        "delta_t": delta_t_final,
        "kalman_filtered": tracking_result.kalman.filtered,
        "kalman_z_scores": tracking_result.kalman.z_scores,
        "cusum_g_plus": tracking_result.cusum.g_plus,
        "cusum_g_minus": tracking_result.cusum.g_minus,
        "cusum_alarms": tracking_result.cusum.alarms,
    })

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path_tracking = export_dir / f"empirical_tracking_{timestamp_str}.parquet"
    export_path_morphology = export_dir / f"empirical_morphology_{timestamp_str}.parquet"

    try:
        df_results.to_parquet(export_path_tracking, index=False)
        print(f"[EXPORT] Tracking saved: {export_path_tracking}")
    except Exception as e:
        df_results.to_csv(export_path_tracking.with_suffix('.csv'), index=False)

    if global_morphology_dfs:
        df_morphology_master = pd.concat(global_morphology_dfs, ignore_index=True)
        try:
            df_morphology_master.to_parquet(export_path_morphology, index=False)
            print(f"[EXPORT] Morphology saved: {export_path_morphology}")
        except Exception:
            pass

    print("[PIPELINE] Empirical pipeline completed.")
    return delta_t_final, tracking_result.kalman, tracking_result.cusum
