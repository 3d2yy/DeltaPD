import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat

from deltapd.campaign.mat_series_study import run_mat_series_study


def _make_synthetic_mat_series() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.0, 0.01, size=(100, 128))
    time = np.linspace(0.0, 127.0, 128)

    pulse_shape = np.array([0.0, 0.12, 0.25, 0.12, 0.0])
    for row_idx in range(40, 50):
        for center in [35, 68, 97]:
            start = center - 2
            matrix[row_idx, start : start + len(pulse_shape)] += pulse_shape

    return matrix, time


def test_run_mat_series_study_end_to_end():
    matrix, time = _make_synthetic_mat_series()

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        mat_path = base / "series.mat"
        cfg_path = base / "series.yaml"
        out_dir = base / "outputs"

        savemat(mat_path, {"serie": matrix, "time": time})
        cfg_path.write_text(
            f"""
input:
  mat_path: "{mat_path.as_posix()}"
  signal_key: "serie"
  time_key: "time"
  fs_hz: 1.0e9
analysis:
  center_rows: true
  threshold_sigma: 4.0
  min_separation_ns: 20.0
  active_quantile: 0.95
  change_top_k: 6
  change_min_row_gap: 5
descriptors:
  features:
    - energy_v2
    - p95_abs_v
    - peak_abs_v
    - active_ratio
    - pulse_count
    - median_dt_s
    - cv_dt
    - event_width_frac
  activity_features:
    - energy_v2
    - p95_abs_v
    - peak_abs_v
    - active_ratio
  change_features:
    - energy_v2
    - p95_abs_v
    - peak_abs_v
    - pulse_count
output_dir: "{out_dir.as_posix()}"
""".strip(),
            encoding="utf-8",
        )

        outputs = run_mat_series_study(cfg_path)

        assert not outputs["row_descriptors"].empty
        assert not outputs["activity_blocks"].empty
        assert not outputs["change_candidates"].empty
        assert any(
            row["row_start"] <= 49 and row["row_end"] >= 40
            for _, row in outputs["activity_blocks"].iterrows()
        )
        assert any(
            35 <= int(row_idx) <= 55
            for row_idx in outputs["change_candidates"]["row_idx"].tolist()
        )
        assert (out_dir / "study_report.md").exists()
        assert (out_dir / "descriptor_trends.png").exists()
        assert (out_dir / "representative_waveforms.png").exists()
