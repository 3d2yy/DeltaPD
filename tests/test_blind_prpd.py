import numpy as np

from deltapd.blind_prpd import (
    calibrate_grid_frequency,
    calibrate_grid_frequency_details,
    compute_local_blind_prpd_trace,
    compare_frequency_estimators,
    reconstruct_blind_prpd,
)


def test_calibrate_grid_frequency_recovers_true_frequency():
    rng = np.random.default_rng(7)
    f_true = 50.1234
    cycles = np.arange(2500, dtype=np.float64)
    phase_choices = rng.choice([30.0, 210.0], size=len(cycles))
    jitter_s = rng.normal(0.0, 4e-6, size=len(cycles))
    toa = cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s
    toa = np.sort(toa)
    peaks = 0.5 + rng.random(len(toa))

    f_est = calibrate_grid_frequency(
        toa,
        base_freq=50.0,
        search_width=0.5,
        peak_weights=peaks,
    )

    assert abs(f_est - f_true) < 1e-3


def test_reconstruct_blind_prpd_aligns_to_latent_bipolar_axis():
    rng = np.random.default_rng(13)
    f_true = 49.978
    cycles = np.arange(1500, dtype=np.float64)
    phase_choices = rng.choice([42.0, 222.0], size=len(cycles))
    jitter_s = rng.normal(0.0, 3e-6, size=len(cycles))
    toa = cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s
    toa = np.sort(toa)
    peaks = 0.8 + 0.4 * rng.random(len(toa))

    phases_deg, peaks_out, info = reconstruct_blind_prpd(
        toa,
        peaks,
        freq_hz=50.0,
        auto_calibrate=True,
        return_metadata=True,
    )

    assert len(phases_deg) == len(toa)
    assert len(peaks_out) == len(toa)
    assert abs(info.calibrated_freq_hz - f_true) < 1e-3
    assert info.requested_method == "auto"
    assert info.selected_method in {"coherence", "harmonic_power", "epoch_folding"}
    assert np.isfinite(info.sharpness)
    assert np.isfinite(info.common_axial_confidence)

    folded = np.mod(phases_deg, 180.0)
    distance_to_axis = np.minimum(folded, 180.0 - folded)
    assert float(np.median(distance_to_axis)) < 10.0


def test_harmonic_power_recovers_frequency_for_narrow_clusters_with_outliers():
    rng = np.random.default_rng(31)
    f_true = 50.081
    cycles = np.arange(1800, dtype=np.float64)
    phase_choices = rng.choice([18.0, 198.0], size=len(cycles))
    jitter_s = rng.normal(0.0, 1.5e-6, size=len(cycles))
    toa = cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s

    outlier_cycles = rng.uniform(0.0, cycles.max() / f_true, size=120)
    toa = np.sort(np.concatenate([toa, outlier_cycles]))
    peaks = np.concatenate([1.0 + 0.2 * rng.random(len(cycles)), 0.2 + 0.1 * rng.random(len(outlier_cycles))])
    peaks = peaks[np.argsort(np.concatenate([cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s, outlier_cycles]))]

    f_est = calibrate_grid_frequency(
        toa,
        base_freq=50.0,
        search_width=0.5,
        peak_weights=peaks,
        method="harmonic_power",
        n_harmonics=4,
    )

    assert abs(f_est - f_true) < 2e-3


def test_compare_frequency_estimators_reports_multiple_methods():
    rng = np.random.default_rng(9)
    f_true = 49.955
    cycles = np.arange(1200, dtype=np.float64)
    phase_choices = rng.choice([35.0, 215.0], size=len(cycles))
    jitter_s = rng.normal(0.0, 2e-6, size=len(cycles))
    toa = np.sort(cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s)
    peaks = 0.5 + rng.random(len(toa))

    results = compare_frequency_estimators(
        toa,
        base_freq=50.0,
        peak_weights=peaks,
        methods=("coherence", "harmonic_power", "epoch_folding"),
        n_harmonics=3,
    )

    assert [row["method"] for row in results] == ["coherence", "harmonic_power", "epoch_folding"]
    assert all(abs(float(row["freq_hz"]) - f_true) < 2e-3 for row in results)
    assert all("sharpness" in row for row in results)
    assert all("selected_method" in row for row in results)
    assert all("common_axial_confidence" in row for row in results)


def test_auto_method_recovers_frequency_on_outlier_heavy_case():
    rng = np.random.default_rng(44)
    f_true = 50.067
    cycles = np.arange(1600, dtype=np.float64)
    phase_choices = rng.choice([20.0, 200.0], size=len(cycles))
    jitter_s = rng.normal(0.0, 1.4e-6, size=len(cycles))
    toa_core = cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s
    peaks_core = 0.9 + 0.2 * rng.random(len(cycles))

    n_outliers = 140
    toa_out = rng.uniform(0.0, cycles.max() / f_true, size=n_outliers)
    peaks_out = 0.15 + 0.08 * rng.random(n_outliers)

    toa = np.concatenate([toa_core, toa_out])
    peaks = np.concatenate([peaks_core, peaks_out])
    order = np.argsort(toa)
    toa = toa[order]
    peaks = peaks[order]

    f_est = calibrate_grid_frequency(
        toa,
        base_freq=50.0,
        search_width=0.5,
        peak_weights=peaks,
        method="auto",
        n_harmonics=4,
    )

    assert abs(f_est - f_true) < 2e-3


def test_bootstrap_stability_metrics_are_exposed_for_auto():
    rng = np.random.default_rng(48)
    f_true = 50.043
    cycles = np.arange(1100, dtype=np.float64)
    phase_choices = rng.choice([24.0, 204.0], size=len(cycles))
    jitter_s = rng.normal(0.0, 2.0e-6, size=len(cycles))
    toa = np.sort(cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s)
    peaks = 0.7 + 0.3 * rng.random(len(toa))

    details = calibrate_grid_frequency_details(
        toa,
        base_freq=50.0,
        search_width=0.5,
        peak_weights=peaks,
        method="auto",
        n_harmonics=4,
        bootstrap_iterations=6,
        bootstrap_sample_fraction=0.75,
        bootstrap_seed=9,
    )

    assert abs(details.freq_hz - f_true) < 2e-3
    assert details.bootstrap_iterations == 6
    assert np.isfinite(details.bootstrap_freq_std_hz)
    assert np.isfinite(details.bootstrap_ci_width_hz)
    assert 0.0 <= details.bootstrap_method_agreement <= 1.0
    assert isinstance(details.bootstrap_selected_method_counts, dict)


def test_reconstruct_blind_prpd_matches_direct_calibration_details():
    rng = np.random.default_rng(52)
    f_true = 50.043
    cycles = np.arange(900, dtype=np.float64)
    phase_choices = rng.choice([22.0, 202.0], p=[0.7, 0.3], size=len(cycles))
    jitter_s = rng.normal(0.0, 1.8e-6, size=len(cycles))
    toa = np.sort(cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s)
    peaks = np.where(phase_choices == 22.0, 1.2 + 0.15 * rng.random(len(cycles)), 0.45 + 0.10 * rng.random(len(cycles)))

    details = calibrate_grid_frequency_details(
        toa,
        base_freq=50.0,
        search_width=0.25,
        coarse_steps=4001,
        peak_weights=peaks,
        method="auto",
        n_harmonics=4,
        bootstrap_iterations=6,
        bootstrap_sample_fraction=0.75,
        bootstrap_seed=42,
    )
    _, _, info = reconstruct_blind_prpd(
        toa,
        peaks,
        freq_hz=50.0,
        auto_calibrate=True,
        calibration_method="auto",
        calibration_search_width_hz=0.25,
        calibration_coarse_steps=4001,
        n_harmonics=4,
        bootstrap_iterations=6,
        bootstrap_sample_fraction=0.75,
        bootstrap_seed=42,
        return_metadata=True,
    )

    assert info.selected_method == details.selected_method
    assert abs(info.calibrated_freq_hz - details.freq_hz) < 1e-12
    assert abs(info.common_axial_confidence - details.common_axial_confidence) < 1e-12


def test_local_window_stability_metrics_are_exposed():
    rng = np.random.default_rng(57)
    f_true = 50.012
    cycles = np.arange(1600, dtype=np.float64)
    phase_choices = rng.choice([28.0, 208.0], p=[0.6, 0.4], size=len(cycles))
    jitter_s = rng.normal(0.0, 2.1e-6, size=len(cycles))
    toa = np.sort(cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s)
    peaks = np.where(phase_choices == 28.0, 1.1 + 0.12 * rng.random(len(cycles)), 0.5 + 0.08 * rng.random(len(cycles)))

    details = calibrate_grid_frequency_details(
        toa,
        base_freq=50.0,
        search_width=0.25,
        coarse_steps=2001,
        peak_weights=peaks,
        method="auto",
        n_harmonics=4,
        local_window_size_events=256,
        local_window_step_events=128,
        local_min_events_per_window=128,
        local_min_window_count=3,
    )

    assert details.local_window_count >= 3
    assert np.isfinite(details.local_freq_std_hz)
    assert np.isfinite(details.local_freq_span_hz)
    assert 0.0 <= details.local_method_agreement <= 1.0


def test_compute_local_blind_prpd_trace_emits_window_rows():
    rng = np.random.default_rng(58)
    f_true = 50.018
    cycles = np.arange(1400, dtype=np.float64)
    phase_choices = rng.choice([30.0, 210.0], p=[0.55, 0.45], size=len(cycles))
    jitter_s = rng.normal(0.0, 2.0e-6, size=len(cycles))
    toa = np.sort(cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s)
    peaks = 0.8 + 0.25 * rng.random(len(toa))

    rows = compute_local_blind_prpd_trace(
        toa,
        base_freq=50.0,
        search_width=0.25,
        coarse_steps=2001,
        peak_weights=peaks,
        method="auto",
        n_harmonics=4,
        window_size_events=256,
        step_events=128,
        min_events_per_window=128,
        min_window_count=3,
        global_freq_hz=f_true,
    )

    assert len(rows) >= 3
    assert rows[0]["local_window_index"] == 0
    assert rows[0]["event_start_idx"] == 0
    assert rows[0]["event_end_idx"] == 255
    assert np.isfinite(rows[0]["freq_hz"])
    assert np.isfinite(rows[0]["freq_offset_from_global_hz"])
    assert rows[0]["selected_method"] in {"coherence", "harmonic_power", "epoch_folding"}


def test_epoch_folding_recovers_frequency_for_clean_axial_case():
    rng = np.random.default_rng(55)
    f_true = 49.931
    cycles = np.arange(1400, dtype=np.float64)
    phase_choices = rng.choice([36.0, 216.0], size=len(cycles))
    jitter_s = rng.normal(0.0, 2.2e-6, size=len(cycles))
    toa = np.sort(cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s)
    peaks = 0.6 + 0.5 * rng.random(len(toa))

    f_est = calibrate_grid_frequency(
        toa,
        base_freq=50.0,
        search_width=0.5,
        peak_weights=peaks,
        method="epoch_folding",
    )

    assert abs(f_est - f_true) < 2e-3


def test_gregory_loredo_recovers_frequency_for_clean_event_folding_case():
    rng = np.random.default_rng(66)
    f_true = 50.027
    cycles = np.arange(1000, dtype=np.float64)
    phase_choices = rng.choice([32.0, 212.0], size=len(cycles))
    jitter_s = rng.normal(0.0, 2.0e-6, size=len(cycles))
    toa = np.sort(cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s)
    peaks = 0.7 + 0.3 * rng.random(len(toa))

    f_est = calibrate_grid_frequency(
        toa,
        base_freq=50.0,
        search_width=0.5,
        peak_weights=peaks,
        method="gregory_loredo",
    )

    assert abs(f_est - f_true) < 3e-3


def test_phase_distance_correlation_recovers_frequency_for_phase_amplitude_structure():
    rng = np.random.default_rng(77)
    f_true = 50.041
    cycles = np.arange(1300, dtype=np.float64)
    phase_choices = rng.choice([24.0, 204.0], p=[0.65, 0.35], size=len(cycles))
    jitter_s = rng.normal(0.0, 1.6e-6, size=len(cycles))
    toa = np.sort(cycles / f_true + (phase_choices / 360.0) / f_true + jitter_s)
    peaks = np.where(phase_choices == 24.0, 1.3 + 0.15 * rng.random(len(cycles)), 0.45 + 0.10 * rng.random(len(cycles)))

    f_est = calibrate_grid_frequency(
        toa,
        base_freq=50.0,
        search_width=0.5,
        peak_weights=peaks,
        method="phase_distance_correlation",
    )

    assert abs(f_est - f_true) < 3e-3
