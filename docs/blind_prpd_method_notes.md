# Blind PRPD Method Notes

## Why this can be a paper contribution

The practical bottleneck is that many UHF/PRPD workflows still rely on an external phase reference, IEC 60270 support, or an additional synchronized sensor chain. A blind method that reconstructs a physically useful phase axis from pulse timestamps alone is therefore publishable if it is:

- statistically explicit;
- compared against strong baselines;
- shown to improve descriptor stability or downstream discrimination.

This repo now treats blind PRPD as an event-time frequency/phase estimation problem on axial circular data.

## Methods to compare

### 1. `coherence`

Current baseline. It maximizes the weighted mean resultant length on the doubled phase:

- data: `phi_i = 4 * pi * f * t_i`;
- score: `|sum(w_i * exp(j * phi_i))| / sum(w_i)`.

This is the most natural Rayleigh-like statistic for bipolar or antipodal phase structure.

### 2. `harmonic_power`

New method. It extends the baseline with several harmonics on the doubled phase:

- score: `sum_h |sum(w_i * exp(j * h * phi_i))|^2 / (h * sum(w_i)^2)`.

This is useful when the axial pattern is narrow, asymmetric, or contaminated by outliers. It is conceptually close to multi-harmonic event-periodicity tests.

### 3. `epoch_folding`

New event-oriented method. It folds the axial phase into a histogram and scores deviation from uniform occupancy, with several sub-bin shifts to reduce bin-edge sensitivity.

This is closer to event epoch-folding statistics than to sinusoidal assumptions, so it is a better third family for UHF pulse timestamps than a direct PDM adaptation.

### 4. `gregory_loredo`

New serious rival. It scores periodicity with a Bayesian event-folding view of unknown-shape phase structure:

- fold events into phase bins;
- average over phase offsets;
- integrate evidence across several bin counts.

Current reading:

- it is a scientifically strong contrast family because it stays in the event-folding world instead of relying on harmonic concentration;
- it behaved much better than `phase_distance_correlation` in both synthetic and real-case checks;
- it did not beat the main block often enough to enter `auto`, but it is strong enough to stay in the paper ablation table.

### 5. `phase_distance_correlation`

New rival family. It scores dependence between folded phase distance and pairwise amplitude distance, following the phase-distance correlation idea from the astronomy literature.

In this repo it is implemented as a practical event-stream adaptation:

- the phase distance is computed on circular folded phase;
- the value distance is built from amplitude-derived event weights;
- the centered distance-correlation score is maximized over trial frequency.

Current reading:

- it is scientifically useful as a contrast family because it does not reduce the problem to first-harmonic concentration;
- it was much weaker than the main families in the current synthetic benchmark and in multichannel downstream tests;
- it should stay as a tested baseline, not as the operational default.

### 6. `auto`

New selector. It calibrates `coherence`, `harmonic_power`, and `epoch_folding`, then chooses the candidate frequency that produces the best weighted axial concentration measured with folded-phase entropy.

This is the method now used in the high-level PRPD reconstruction path.

## Calibration diagnostics now exposed

The blind PRPD path now exports not only the calibrated frequency, but also:

- `requested_method`
- `selected_method`
- `coherence`
- `axial_entropy_score`
- `sharpness`
- `half_height_width_hz`
- `common_axial_peak_freq_hz`
- `common_axial_peak_offset_hz`
- `common_axial_confidence`
- `bootstrap_freq_std_hz`
- `bootstrap_ci_width_hz`
- `bootstrap_method_agreement`
- `candidate_spread_hz`
- `winner_margin`

These diagnostics now propagate to:

- `run_manifest.json`
- descriptor-study markdown and PDF reports
- state/alarm batch summaries
- the visual workbench

## How to read the new uncertainty metrics

Primary outputs:

- `outputs/comparative_ch3/comparative_summary.md`
- `outputs/state_alarm_ch3/state_alarm_batch_summary.md`
- `outputs/blind_prpd_real_cases_gl/blind_prpd_real_case_comparison.md`

Reading rule:

- `common_axial_confidence` measures how convincing the shared axial peak is around the winning frequency. High values mean the local axial profile is sharp and prominent relative to its width.
- `common_axial_peak_offset_hz` measures how far the chosen frequency is from the common axial peak. Near-zero values mean the winner sits on the same local optimum seen by the normalized axial profile.
- `bootstrap_freq_std_hz` measures how much the calibrated frequency moves under event resampling. This is the main frequency-stability number.
- `bootstrap_method_agreement` measures how often the bootstrap reruns keep the same winning family. This is the main selector-stability number.

Current CH3 reading:

- `P2` is frequency-stable (`boot std ~ 0.0006 Hz`) but method-ambiguous (`boot agree ~ 0.17`): the frequency peak is clear, but several families land on almost the same optimum.
- `G1` behaves similarly (`boot std ~ 0.0003 Hz`, `boot agree ~ 0.33`): stable frequency, mild family rivalry.
- `P1` is the unstable benchmark case (`boot std ~ 0.0743 Hz`): the downstream descriptors still work, but the frequency calibration itself is much less rigid.
- `P3` and `G3` are the important multiple-source cases: the frequency peak is narrow (`~ 0.0027 Hz` and `~ 0.0003 Hz`) but method agreement is only moderate (`~ 0.67` and `~ 0.50`), which is exactly the kind of ambiguity a good paper should report instead of hiding.
- `G2` combines low common confidence with wider bootstrap spread (`boot std ~ 0.0050 Hz`), so it is the strongest warning case among the superficials.

Interpretation for the paper:

- low `boot std` with low `boot agree` means the frequency is stable even if the selector family is not unique;
- high `boot std` means the calibration itself is fragile and should not be oversold;
- low `common_axial_confidence` means the chosen peak exists, but the axial evidence around it is weak or broad;
- the three numbers should be read together, not in isolation.

## Current synthetic benchmark

Primary outputs:

- `outputs/blind_prpd_benchmark/blind_prpd_benchmark_detail.csv`
- `outputs/blind_prpd_benchmark/blind_prpd_benchmark_summary.csv`
- `outputs/blind_prpd_benchmark/blind_prpd_benchmark_summary.md`

Serious-rival benchmark used for the latest method decision:

- `outputs/blind_prpd_benchmark_gl/blind_prpd_benchmark_summary.md`
- `outputs/blind_prpd_real_cases_gl/blind_prpd_real_case_comparison.md`

Scenarios:

- `symmetric_moderate`
- `narrow_outliers`
- `asymmetric_polarity`
- `missing_cycles`

Current reading:

- `coherence` remains excellent for balanced or dropout-dominated cases;
- `harmonic_power` is better when clusters are narrow or one polarity dominates;
- `epoch_folding` gives a non-sinusoidal event-folding baseline that is often useful as an arbiter;
- `gregory_loredo` is a serious rival and a paper-worthy baseline, but it does not dethrone the main operational block;
- `phase_distance_correlation` is interesting as a dependency-based rival but was clearly weaker in the current synthetic and downstream comparisons;
- `auto` is the safest high-level choice because it preserves the strong baseline and only switches when the axial concentration improves.

## Stress benchmark with drift and gaps

Outputs:

- `outputs/blind_prpd_stress/blind_prpd_stress_summary.csv`
- `outputs/blind_prpd_stress/blind_prpd_stress_summary.md`
- `outputs/blind_prpd_stress/blind_prpd_stress_phase_error_heatmap.png`
- `outputs/blind_prpd_stress/blind_prpd_stress_frequency_error_heatmap.png`

Current reading:

- the key metric here is axial phase error, not only frequency error;
- `coherence` was the best method in four of five stress scenarios when the target was phase preservation under a constant-frequency approximation;
- `harmonic_power` edged the strongest pure linear-drift case;
- `gregory_loredo` often improved scalar frequency error but did not preserve phase structure well enough to justify entry into `auto`;
- `epoch_folding` and `auto` degraded more under segmented gaps than the simpler circular-concentration block;
- `sharpness` should be interpreted within each method family, not compared directly across families, because the score scales are different.

## What to test next for the paper

1. Re-run `P1/P2/P3/G1/G2/G3` with fixed `coherence`, `harmonic_power`, `epoch_folding`, `gregory_loredo`, and `auto`, then compare:
   - phase entropy
   - phase width
   - Kuramoto `R`
   - descriptor-study scores for `type`, `state`, and `alarm`
2. Add long-acquisition stress tests with:
   - frequency drift
   - segmented gaps
   - missing half-cycles
3. Quantify uncertainty:
   - contiguous-window stability on long captures
   - entropy gain of `auto` vs `coherence`
   - whether local frequency bands stay narrow when `boot agree` is low
4. Decide whether `gregory_loredo` should remain a fixed ablation-only rival or earn a conditional place in a future `auto` variant for multiple-source stress cases

## Sources

- Gregory & Loredo, periodic signals of unknown shape in event data:
  [NASA NTRS](https://ntrs.nasa.gov/citations/19930035792)
- Lomb, least-squares frequency analysis for unevenly spaced data:
  [ScienceDirect](https://www.sciencedirect.com/science/article/pii/0012821X76900504)
- Stellingwerf, phase dispersion minimization:
  [NASA NTRS](https://ntrs.nasa.gov/citations/19790030137)
- Zechmeister & Kurster, generalized Lomb-Scargle periodograms:
  [A&A](https://www.aanda.org/articles/aa/abs/2009/11/aa11296-08/aa11296-08.html)
- Leahy et al., epoch folding and periodicity search in event data:
  [ADS](https://ui.adsabs.harvard.edu/abs/1983ApJ...266..160L/abstract)
- Efron, bootstrap resampling:
  [Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/AOS/1176344552)
- Zucker, phase distance correlation periodogram:
  [arXiv 1710.10713](https://arxiv.org/abs/1710.10713)
- Phase-distance correlation periodograms:
  [A&A](https://www.aanda.org/articles/aa/abs/2024/06/aa47764-23/aa47764-23.html)
- Example of UHF/PRPD workflows still tied to external phase information:
  [MDPI Sensors 2024](https://www.mdpi.com/1424-8220/24/13/4136)
  and [MDPI Applied Sciences 2024](https://www.mdpi.com/2076-3417/14/3/1208)
