# Assumptions & Limitations

This document describes the assumptions, simplifications, and limitations of the DeltaPD framework. It is intended to provide transparency for reviewers and users.

## Signal Model

- **Pulse shape**: The synthetic signal generator uses a double-exponential (Gemant-Philippoff) model. Real PD pulses vary in morphology depending on insulation geometry, defect type, and aging.
- **Dielectric channel**: The transfer function `H_d(f)` assumes homogeneous permittivity. Stratified or non-linear dielectrics are not modelled.
- **Antenna response**: The Vivaldi antipodal model is a 4th-order bandpass approximation. Measured S-parameters from a real antenna would improve fidelity.
- **Noise model**: AWGN + narrowband interference (NBI) + corona noise. Real substation EMI includes impulse noise, power-line harmonics, and other sources not covered here.

## Wavelet Optimisation (Phase 1)

- The Monte Carlo grid search is a **heuristic**, not a globally optimal search. The result depends on the grid (wavelet families, threshold rules) and the number of iterations.
- The variance criterion `Var[RMSE] < epsilon` uses a fixed epsilon. No automatic epsilon selection is provided.
- The "feasibility rate" metric is informational; a low rate does not necessarily indicate a problem with the method.

## Delta-t Extraction (Phase 2)

- Pulse detection depends on a user-specified threshold (multiples of sigma). There is no automatic threshold selection.
- The minimum inter-pulse separation is set to zero by default, which may cause false positives in high-noise conditions.
- The framework deliberately discards amplitude information. If amplitude features are relevant to your research, this pipeline is not appropriate.

## Tracking Algorithms (Phase 3)

- The Kalman filter assumes a random-walk state model with constant process/measurement noise variances. These are user-specified and not estimated from data.
- CUSUM threshold and drift are hyperparameters. No automatic tuning or cross-validation is provided.
- The EWMA adaptation rule clips alpha to [0.05, 0.8]. This range was chosen empirically and may not be optimal for all signals.

## Big-O Estimation (Phase 4)

- Complexity is measured via **wall-clock timing**, which is influenced by OS scheduling, cache effects, and background processes.
- The power-law fit `t(n) = a * n^b` is a simplistic model. Real algorithmic complexity may not follow a clean power law at small n.
- Results should be treated as approximate heuristics, not formal complexity proofs.

## Baseline Comparisons

- The energy, ZCR, and Hilbert envelope detectors are simplified implementations. Production-grade implementations may use more sophisticated windowing or adaptive thresholds.
- Comparisons are performed on synthetic data only. Performance on real data may differ significantly.

## ROC and Sensitivity Analysis

- ROC curves are computed by sweeping CUSUM thresholds, not by varying a true positive probability. This is a proxy metric.
- SNR sensitivity uses the same signal model for generation and evaluation. Generalisation to other signal models is not guaranteed.

## Data Loader

- `.mat` files are read via `scipy.io.loadmat`, which supports MATLAB v5-v7.2 files. MATLAB v7.3+ (HDF5-based `.mat`) requires `h5py`.
- Sampling frequency is inferred heuristically from file metadata. Always verify `fs` when loading external data.
