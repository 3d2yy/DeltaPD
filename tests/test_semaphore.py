import pandas as pd

from deltapd.semaphore import build_semaphore_df


def test_build_semaphore_df_defaults_missing_ingestion_to_gray():
    rows = [
        {
            "dataset_key": "A",
            "max_abs_local_freq_offset_hz": 0.20,
            "local_freq_offset_std_hz": 0.10,
            "local_common_axial_confidence_std": 0.20,
            "transition_method_entropy": 0.40,
            "local_regime_transition_entropy": 0.30,
            "local_method_switch_rate": 0.20,
            "mean_local_common_axial_confidence": 0.60,
            "local_regime_mean_run_length": 2.0,
            "n_transition_windows": 6,
        },
        {
            "dataset_key": "B",
            "max_abs_local_freq_offset_hz": 0.40,
            "local_freq_offset_std_hz": 0.20,
            "local_common_axial_confidence_std": 0.40,
            "transition_method_entropy": 0.60,
            "local_regime_transition_entropy": 0.40,
            "local_method_switch_rate": 0.30,
            "mean_local_common_axial_confidence": 0.50,
            "local_regime_mean_run_length": 1.0,
            "n_transition_windows": 6,
        },
    ]

    df = build_semaphore_df(rows)

    assert set(df["semaphore_band"]) == {"gray"}
    assert (df["semaphore_top_drivers"].str.contains("low_ingestion_confidence")).all()


def test_build_semaphore_df_dampens_short_transition_sequences():
    rows = [
        {
            "dataset_key": "control",
            "ingestion_confidence": 1.0,
            "max_abs_local_freq_offset_hz": 0.10,
            "local_freq_offset_std_hz": 0.05,
            "local_common_axial_confidence_std": 0.10,
            "transition_method_entropy": 0.10,
            "local_regime_transition_entropy": 0.10,
            "local_method_switch_rate": 0.10,
            "mean_local_common_axial_confidence": 0.70,
            "local_regime_mean_run_length": 2.0,
            "n_transition_windows": 8,
        },
        {
            "dataset_key": "short",
            "ingestion_confidence": 1.0,
            "max_abs_local_freq_offset_hz": 0.20,
            "local_freq_offset_std_hz": 0.10,
            "local_common_axial_confidence_std": 0.20,
            "transition_method_entropy": 0.90,
            "local_regime_transition_entropy": 0.90,
            "local_method_switch_rate": 0.90,
            "mean_local_common_axial_confidence": 0.60,
            "local_regime_mean_run_length": 1.0,
            "n_transition_windows": 3,
        },
        {
            "dataset_key": "long",
            "ingestion_confidence": 1.0,
            "max_abs_local_freq_offset_hz": 0.25,
            "local_freq_offset_std_hz": 0.12,
            "local_common_axial_confidence_std": 0.22,
            "transition_method_entropy": 0.90,
            "local_regime_transition_entropy": 0.90,
            "local_method_switch_rate": 0.90,
            "mean_local_common_axial_confidence": 0.60,
            "local_regime_mean_run_length": 1.0,
            "n_transition_windows": 8,
        },
    ]

    df = build_semaphore_df(rows).set_index("dataset_key")

    assert df.loc["short", "semaphore_transition_evidence"] < df.loc["long", "semaphore_transition_evidence"]
    assert df.loc["short", "semaphore_risk_score"] < df.loc["long", "semaphore_risk_score"]


def test_build_semaphore_df_uses_bocpd_change_evidence():
    rows = [
        {
            "dataset_key": "stable",
            "ingestion_confidence": 1.0,
            "max_abs_local_freq_offset_hz": 0.18,
            "local_freq_offset_std_hz": 0.08,
            "local_common_axial_confidence_std": 0.12,
            "transition_method_entropy": 0.35,
            "local_regime_transition_entropy": 0.20,
            "local_method_switch_rate": 0.15,
            "mean_local_common_axial_confidence": 0.68,
            "local_regime_mean_run_length": 2.8,
            "bocpd_max_change_prob": 0.12,
            "bocpd_surprise_score": 0.20,
            "bocpd_run_length_mean": 2.5,
            "n_transition_windows": 6,
        },
        {
            "dataset_key": "unstable",
            "ingestion_confidence": 1.0,
            "max_abs_local_freq_offset_hz": 0.18,
            "local_freq_offset_std_hz": 0.08,
            "local_common_axial_confidence_std": 0.12,
            "transition_method_entropy": 0.35,
            "local_regime_transition_entropy": 0.20,
            "local_method_switch_rate": 0.15,
            "mean_local_common_axial_confidence": 0.68,
            "local_regime_mean_run_length": 2.8,
            "bocpd_max_change_prob": 0.82,
            "bocpd_surprise_score": 1.30,
            "bocpd_run_length_mean": 0.9,
            "n_transition_windows": 6,
        },
    ]

    df = build_semaphore_df(rows).set_index("dataset_key")

    assert df.loc["unstable", "semaphore_risk_score"] > df.loc["stable", "semaphore_risk_score"]


def test_build_semaphore_df_uses_hmm_sequence_evidence():
    rows = [
        {
            "dataset_key": "stable",
            "ingestion_confidence": 1.0,
            "max_abs_local_freq_offset_hz": 0.18,
            "local_freq_offset_std_hz": 0.08,
            "local_common_axial_confidence_std": 0.12,
            "transition_method_entropy": 0.35,
            "local_regime_transition_entropy": 0.20,
            "local_method_switch_rate": 0.15,
            "mean_local_common_axial_confidence": 0.68,
            "local_regime_mean_run_length": 2.8,
            "hmm_high_state_share": 0.15,
            "hmm_state_switch_rate": 0.10,
            "hmm_state_entropy": 0.18,
            "hmm_state_mean_run_length": 3.0,
            "n_transition_windows": 6,
        },
        {
            "dataset_key": "unstable",
            "ingestion_confidence": 1.0,
            "max_abs_local_freq_offset_hz": 0.18,
            "local_freq_offset_std_hz": 0.08,
            "local_common_axial_confidence_std": 0.12,
            "transition_method_entropy": 0.35,
            "local_regime_transition_entropy": 0.20,
            "local_method_switch_rate": 0.15,
            "mean_local_common_axial_confidence": 0.68,
            "local_regime_mean_run_length": 2.8,
            "hmm_high_state_share": 0.82,
            "hmm_state_switch_rate": 0.80,
            "hmm_state_entropy": 0.92,
            "hmm_state_mean_run_length": 1.1,
            "n_transition_windows": 6,
        },
    ]

    df = build_semaphore_df(rows).set_index("dataset_key")

    assert df.loc["unstable", "semaphore_risk_score"] > df.loc["stable", "semaphore_risk_score"]


def test_build_semaphore_df_uses_semi_markov_sequence_evidence():
    rows = [
        {
            "dataset_key": "stable",
            "ingestion_confidence": 1.0,
            "max_abs_local_freq_offset_hz": 0.18,
            "local_freq_offset_std_hz": 0.08,
            "local_common_axial_confidence_std": 0.12,
            "transition_method_entropy": 0.35,
            "local_regime_transition_entropy": 0.20,
            "local_method_switch_rate": 0.15,
            "mean_local_common_axial_confidence": 0.68,
            "local_regime_mean_run_length": 2.8,
            "semi_markov_high_state_share": 0.18,
            "semi_markov_state_switch_rate": 0.10,
            "semi_markov_state_entropy": 0.20,
            "semi_markov_state_mean_run_length": 3.2,
            "n_transition_windows": 6,
        },
        {
            "dataset_key": "unstable",
            "ingestion_confidence": 1.0,
            "max_abs_local_freq_offset_hz": 0.18,
            "local_freq_offset_std_hz": 0.08,
            "local_common_axial_confidence_std": 0.12,
            "transition_method_entropy": 0.35,
            "local_regime_transition_entropy": 0.20,
            "local_method_switch_rate": 0.15,
            "mean_local_common_axial_confidence": 0.68,
            "local_regime_mean_run_length": 2.8,
            "semi_markov_high_state_share": 0.84,
            "semi_markov_state_switch_rate": 0.82,
            "semi_markov_state_entropy": 0.94,
            "semi_markov_state_mean_run_length": 1.1,
            "n_transition_windows": 6,
        },
    ]

    df = build_semaphore_df(rows).set_index("dataset_key")

    assert df.loc["unstable", "semaphore_risk_score"] > df.loc["stable", "semaphore_risk_score"]
