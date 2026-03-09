import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from deltapd.campaign.descriptor_study import (
    PRIMARY_DESCRIPTOR_BANK,
    _build_blind_transition_overlap,
    build_feature_windows,
    run_descriptor_study,
)


def _make_synthetic_event_table() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    toa = 0.0
    event_idx = 1

    stage_params = {
        1: {"dt_mean": 8.0e-4, "dt_std": 1.2e-4, "phase_std": 8.0, "peak": 0.08},
        2: {"dt_mean": 4.5e-4, "dt_std": 1.5e-4, "phase_std": 18.0, "peak": 0.12},
        3: {"dt_mean": 1.7e-4, "dt_std": 7.0e-5, "phase_std": 32.0, "peak": 0.18},
    }

    for stage, cfg in stage_params.items():
        for i in range(120):
            dt = max(1e-8, rng.normal(cfg["dt_mean"], cfg["dt_std"]))
            toa += dt
            center = 70.0 if i % 2 == 0 else 250.0
            phase = np.mod(rng.normal(center, cfg["phase_std"]), 360.0)
            peak = abs(rng.normal(cfg["peak"], cfg["peak"] * 0.15))
            rows.append(
                {
                    "event_idx": event_idx,
                    "toa_s": toa,
                    "delta_t_s": dt,
                    "log10_dt": np.log10(dt),
                    "pulse_rate_hz": 1.0 / dt,
                    "peak_v": peak,
                    "prpd_phase_deg": phase,
                    "is_outlier": False,
                    "stage": stage,
                }
            )
            event_idx += 1

    return pd.DataFrame(rows)


def test_build_feature_windows_generates_descriptor_matrix():
    df_events = _make_synthetic_event_table()
    df_windows = build_feature_windows(
        df_events,
        window_events=32,
        step_events=16,
        min_valid_events=16,
    )

    assert not df_windows.empty
    for feature in PRIMARY_DESCRIPTOR_BANK:
        assert feature in df_windows.columns
    for feature in [
        "phase_neg_mean_deg",
        "phase_neg_q25_deg",
        "phase_neg_count",
        "phase_pos_mean_deg",
        "phase_pos_q75_deg",
        "phase_pos_count",
    ]:
        assert feature in df_windows.columns
    assert "stage" in df_windows.columns


def test_run_descriptor_study_end_to_end():
    df_events = _make_synthetic_event_table()

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        event_csv = base / "delta_t_series_master.csv"
        cfg_path = base / "study.yaml"
        out_dir = base / "outputs"

        df_events.to_csv(event_csv, index=False)
        cfg_path.write_text(
            f"""
input:
  event_csv: "{event_csv.as_posix()}"
windowing:
  window_events: 32
  step_events: 16
  min_valid_events: 16
  max_valid_dt_s: 1.0
  fano_bin_count: 6
descriptors:
  search_features:
    - median_dt_s
    - iqr_dt_s
    - cv_dt
    - cv2_dt
    - weibull_beta
    - burstiness
    - fano_factor
    - phase_entropy
    - phase_kuramoto_r
    - phase_width_pos_deg
  reserve_features:
    - phase_width_neg_deg
    - phase_inlier_ratio
    - mean_peak_v
tasks:
  state:
    type: "multiclass"
    label_column: "stage"
  alarm:
    type: "binary"
    label_column: "stage"
    positive_values: [3]
search:
  n_splits: 3
  random_seed: 42
  top_k_features: 6
  max_combo_size: 2
  forward_selection_max_features: 3
  score_tolerance: 0.01
  redundancy_threshold: 0.85
output_dir: "{out_dir.as_posix()}"
""".strip(),
            encoding="utf-8",
        )

        outputs = run_descriptor_study(cfg_path)

        assert not outputs["windows"].empty
        assert not outputs["state_univariate"].empty
        assert not outputs["alarm_univariate"].empty
        assert "state" in outputs["recommendations"]
        assert "alarm" in outputs["recommendations"]
        assert (out_dir / "study_report.md").exists()
        assert (out_dir / "state_univariate.csv").exists()
        assert (out_dir / "alarm_exhaustive.csv").exists()


def test_run_descriptor_study_falls_back_to_time_segments_for_degenerate_labels():
    df_events = _make_synthetic_event_table()
    df_events["stage"] = 1

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        event_csv = base / "delta_t_series_master.csv"
        cfg_path = base / "study.yaml"
        out_dir = base / "outputs"

        df_events.to_csv(event_csv, index=False)
        cfg_path.write_text(
            f"""
input:
  event_csv: "{event_csv.as_posix()}"
windowing:
  window_events: 32
  step_events: 16
  min_valid_events: 16
descriptors:
  search_features:
    - median_dt_s
    - iqr_dt_s
    - cv_dt
    - burstiness
    - phase_entropy
    - phase_kuramoto_r
tasks:
  state:
    type: "multiclass"
    label_column: "stage"
    fallback_time_segments: 3
  alarm:
    type: "binary"
    label_column: "stage"
    fallback_time_segments: 3
search:
  n_splits: 3
  random_seed: 42
  top_k_features: 4
  max_combo_size: 2
  forward_selection_max_features: 3
  score_tolerance: 0.01
  redundancy_threshold: 0.85
output_dir: "{out_dir.as_posix()}"
""".strip(),
            encoding="utf-8",
        )

        outputs = run_descriptor_study(cfg_path)
        report_text = (out_dir / "study_report.md").read_text(encoding="utf-8")

        assert "__auto_state" in outputs["windows"].columns
        assert "__auto_alarm" in outputs["windows"].columns
        assert outputs["state_univariate"]["n_classes"].max() >= 3
        assert outputs["alarm_univariate"]["n_classes"].max() == 2
        assert "automatic time segmentation" in report_text


def test_run_descriptor_study_exports_blind_transition_overlap():
    df_events = _make_synthetic_event_table()

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        campaign_dir = base / "campaign"
        campaign_dir.mkdir(parents=True, exist_ok=True)
        material_out_dir = base / "material_out"
        material_out_dir.mkdir(parents=True, exist_ok=True)
        material_cfg_path = campaign_dir / "material.yaml"
        study_cfg_path = campaign_dir / "study.yaml"
        out_dir = base / "outputs"

        df_events.to_csv(material_out_dir / "delta_t_series_master.csv", index=False)
        pd.DataFrame(
            [
                {
                    "local_window_index": 0,
                    "event_start_idx": 0,
                    "event_end_idx": 63,
                    "toa_start_s": float(df_events["toa_s"].iloc[0]),
                    "toa_end_s": float(df_events["toa_s"].iloc[63]),
                    "toa_center_s": float(
                        0.5 * (df_events["toa_s"].iloc[0] + df_events["toa_s"].iloc[63])
                    ),
                    "n_events": 64,
                    "selected_method": "coherence",
                    "freq_hz": 50.01,
                    "freq_offset_from_global_hz": 0.002,
                    "coherence": 0.91,
                    "common_axial_confidence": 0.87,
                },
                {
                    "local_window_index": 1,
                    "event_start_idx": 64,
                    "event_end_idx": 127,
                    "toa_start_s": float(df_events["toa_s"].iloc[64]),
                    "toa_end_s": float(df_events["toa_s"].iloc[127]),
                    "toa_center_s": float(
                        0.5 * (df_events["toa_s"].iloc[64] + df_events["toa_s"].iloc[127])
                    ),
                    "n_events": 64,
                    "selected_method": "harmonic_power",
                    "freq_hz": 49.99,
                    "freq_offset_from_global_hz": -0.003,
                    "coherence": 0.88,
                    "common_axial_confidence": 0.82,
                },
            ]
        ).to_csv(material_out_dir / "blind_prpd_local_trace.csv", index=False)
        (material_out_dir / "run_manifest.json").write_text(
            '{"blind_prpd":{"requested_method":"auto","selected_method":"coherence","calibrated_freq_hz":50.0}}',
            encoding="utf-8",
        )
        material_cfg_path.write_text(
            yaml.safe_dump(
                {
                    "output_dir": str(material_out_dir),
                    "dataset": {"folder": ".", "channel": "CH3"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        study_cfg_path.write_text(
            f"""
material_config: "{material_cfg_path.as_posix()}"
windowing:
  window_events: 32
  step_events: 16
  min_valid_events: 16
descriptors:
  search_features:
    - median_dt_s
    - iqr_dt_s
    - cv_dt
    - burstiness
    - phase_entropy
    - phase_kuramoto_r
tasks:
  state:
    type: "multiclass"
    label_column: "stage"
  alarm:
    type: "binary"
    label_column: "stage"
    positive_values: [3]
change_detection:
  enabled: true
  top_k: 4
  min_window_gap: 1
search:
  n_splits: 3
  random_seed: 42
  top_k_features: 4
  max_combo_size: 2
  forward_selection_max_features: 3
  score_tolerance: 0.01
  redundancy_threshold: 0.85
output_dir: "{out_dir.as_posix()}"
""".strip(),
            encoding="utf-8",
        )

        outputs = run_descriptor_study(study_cfg_path)
        report_text = (out_dir / "study_report.md").read_text(encoding="utf-8")

        assert not outputs["blind_transition_overlap"].empty
        assert (out_dir / "blind_prpd_transition_overlap.csv").exists()
        assert (out_dir / "blind_prpd_transition_map.png").exists()
        assert "Blind PRPD / Transition Overlap" in report_text


def test_build_blind_transition_overlap_marks_primary_local_matches():
    change_candidates_df = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "toa_start_s": 0.0,
                "toa_end_s": 1.0,
                "change_score": 2.4,
                "dominant_feature": "iqr_dt_s",
                "dominant_delta_z": 1.1,
            },
            {
                "candidate_rank": 2,
                "toa_start_s": 0.2,
                "toa_end_s": 1.1,
                "change_score": 2.1,
                "dominant_feature": "phase_entropy",
                "dominant_delta_z": 0.9,
            },
            {
                "candidate_rank": 3,
                "toa_start_s": 2.0,
                "toa_end_s": 3.0,
                "change_score": 1.8,
                "dominant_feature": "cv_dt",
                "dominant_delta_z": 0.7,
            },
        ]
    )
    blind_trace_df = pd.DataFrame(
        [
            {
                "local_window_index": 0,
                "toa_start_s": 0.0,
                "toa_end_s": 1.2,
                "toa_center_s": 0.6,
                "selected_method": "coherence",
                "freq_hz": 50.01,
                "freq_offset_from_global_hz": 0.01,
                "common_axial_confidence": 0.82,
                "coherence": 0.91,
                "n_events": 128,
            },
            {
                "local_window_index": 1,
                "toa_start_s": 2.0,
                "toa_end_s": 3.0,
                "toa_center_s": 2.5,
                "selected_method": "harmonic_power",
                "freq_hz": 49.99,
                "freq_offset_from_global_hz": -0.01,
                "common_axial_confidence": 0.78,
                "coherence": 0.88,
                "n_events": 128,
            },
        ]
    )

    overlap_df = _build_blind_transition_overlap(change_candidates_df, blind_trace_df)

    assert overlap_df["local_window_candidate_count"].tolist() == [2, 2, 1]
    assert overlap_df["local_window_match_rank"].tolist() == [1, 2, 1]
    assert overlap_df["is_primary_local_match"].tolist() == [True, False, True]
