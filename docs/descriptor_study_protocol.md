# Descriptor Study Protocol

## Objective

This protocol turns the material-state pipeline into a reproducible study for:

- `alarm`: binary discrimination between nominal and degraded behavior
- `state`: multi-stage discrimination across degradation regimes
- optional secondary tasks such as source/type classification if labels exist

The main scientific question is not "which descriptor looks nice", but which descriptors are:

- individually informative,
- complementary in combination,
- robust to windowing and preprocessing choices,
- simple enough to deploy in an alarm layer.

## Input

The study expects an event-level table such as:

- `outputs/material_state_outputs/delta_t_series_master.csv`

This table is already produced by the material-state pipeline and contains:

- `toa_s`
- `delta_t_s`
- `peak_v`
- `prpd_phase_deg`
- `is_outlier`
- `stage` (if stage-aware mode is enabled)

## Windowing

Descriptors are computed over sliding windows of events, not over fixed time spans.

Recommended default:

- `window_events = 64`
- `step_events = 16`
- `min_valid_events = 32`

This gives smoother descriptor trajectories and reduces dependence on sparse local pulse counts.

## Descriptor Bank

Primary descriptors:

- `median_dt_s`
- `iqr_dt_s`
- `p90_dt_s`
- `cv_dt`
- `cv2_dt`
- `local_variation`
- `weibull_beta`
- `burstiness`
- `fano_factor`
- `phase_entropy`
- `phase_kuramoto_r`
- `phase_width_pos_deg`
- `phase_width_neg_deg`

Reserve / baseline descriptors:

- `phase_inlier_ratio`
- `amplitude_balance_ratio`
- `mean_peak_v`
- `n_events`

## Evaluation Protocol

Each descriptor is first evaluated alone.

Then the study runs:

- exhaustive search for combinations of size `2..max_combo_size`
- greedy forward selection up to `forward_selection_max_features`
- Spearman redundancy analysis

## Metrics

Binary tasks:

- `AUROC`
- `balanced_accuracy`
- `precision`
- `recall`
- `F1`

Multiclass tasks:

- `macro_f1`
- `balanced_accuracy`

## Selection Rule

The recommended subset is the smallest subset within a configurable tolerance of the best score.

This keeps the final alarm model compact and avoids bloated descriptor banks that only look better because they are larger.

## Recommended Workflow

1. Run `run-material` with stage-aware labels enabled.
2. Run `run-study` on the resulting event table.
3. Compare:
   - univariate rankings,
   - exhaustive combinations,
   - forward-selection path,
   - redundancy pairs.
4. Keep the smallest stable subset for the alarm model.
