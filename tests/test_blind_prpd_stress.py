from pathlib import Path

import pandas as pd

from deltapd.blind_prpd_stress import (
    _axial_phase_error_summary,
    run_blind_prpd_stress_benchmark,
)


def test_axial_phase_error_summary_is_zero_for_exact_global_shift():
    toa_s = [0.0, 0.01, 0.02, 0.03]
    true_phase_deg = [20.0, 200.0, 20.0, 200.0]
    peaks = [1.0, 1.0, 1.0, 1.0]
    estimated_freq_hz = 50.0

    # These times correspond to phases shifted by +17 deg relative to truth.
    shifted_toa = [(phase + 17.0) / (360.0 * estimated_freq_hz) + idx / estimated_freq_hz for idx, phase in enumerate([20.0, 200.0, 20.0, 200.0])]
    summary = _axial_phase_error_summary(
        toa_s=shifted_toa,
        true_phase_deg=true_phase_deg,
        peaks=peaks,
        estimated_freq_hz=estimated_freq_hz,
    )

    assert summary["mean_axial_phase_error_deg"] < 1e-9
    assert summary["p95_axial_phase_error_deg"] < 1e-9


def test_run_blind_prpd_stress_benchmark_writes_outputs(tmp_path: Path):
    outputs = run_blind_prpd_stress_benchmark(
        output_dir=tmp_path,
        scenarios=("linear_drift_mild", "segmented_gaps"),
        n_trials=1,
        seed=9,
        cycle_range=(120, 140),
        search_width=0.35,
    )

    for path in outputs.values():
        assert Path(path).exists()

    summary_df = pd.read_csv(outputs["summary_csv"])
    assert set(summary_df["scenario"]) == {"linear_drift_mild", "segmented_gaps"}
    assert {"coherence", "harmonic_power", "epoch_folding", "h_test", "pdm", "gregory_loredo", "auto"} <= set(summary_df["method"])
    assert "mean_axial_phase_error_deg" in summary_df.columns
