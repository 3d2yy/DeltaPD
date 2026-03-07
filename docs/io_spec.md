# I/O Specification

Detailed input/output contracts for each pipeline phase.

## Phase 1 — Wavelet Optimisation

**Input:**
- `n_samples`: int — Number of signal samples (default: 4096)
- `fs`: float — Sampling frequency in Hz (default: 1e9)
- `n_iterations`: int — Monte Carlo iterations (default: 500)
- `seed`: int — Random seed for reproducibility

**Output:** `MonteCarloResult` dataclass
- `best_wavelet`: str — e.g. `"db4"`
- `best_threshold_mode`: str — `"soft"` or `"hard"`
- `best_threshold_rule`: str — `"universal"`, `"minimax"`, `"sqtwolog"`, or `"sure"`
- `best_rmse_mean`: float — E[RMSE] of the optimal configuration
- `best_rmse_var`: float — Var[RMSE]
- `grid`: list[WaveletGridPoint] — Full search grid
- `n_feasible`: int — Number of configurations meeting variance criterion
- `feasibility_rate`: float — Fraction of feasible configurations
- `converged`: bool — Whether any configuration met the variance criterion

## Phase 2 — Delta-t Extraction

**Input:**
- `signal_data`: ndarray, shape `(n_samples,)`, dtype float64 — Denoised signal
- `fs`: float — Sampling frequency in Hz

**Output:**
- `delta_t`: ndarray, shape `(n_pulses - 1,)`, dtype float64 — Inter-pulse intervals in **seconds**
- `pulse_indices`: ndarray of int — Sample indices of detected pulses

**Invariants:**
- All delta-t values are strictly positive
- `len(delta_t) == len(pulse_indices) - 1`

## Phase 3 — Tracking

**Input:**
- `delta_t`: ndarray, shape `(N,)` — Inter-pulse intervals

**Output:** `DeltaTTrackingResult` dataclass containing:

| Algorithm | Key fields | Shapes |
|-----------|-----------|--------|
| Kalman | `filtered`, `residuals`, `kalman_gains`, `steady_state_gain` | `(N,)`, `(N,)`, `(N,)`, scalar |
| EWMA | `smoothed`, `residuals`, `alpha_sequence` | `(N,)`, `(N,)`, `(N,)` |
| CUSUM | `g_plus`, `g_minus`, `alarms`, `alarm_indices`, `n_alarms` | `(N,)`, `(N,)`, `(N,)` bool, `(k,)` int, scalar |

## Phase 4 — Quantification

**Input:**
- `sizes`: tuple of int — Input sizes for complexity measurement
- `n_repeats`: int — Repetitions per size

**Output:**
- `BigOEstimate` per algorithm: `exponent_b`, `coefficient_a`, `r_squared`, `big_o_label`
- `ConvergenceConfusionMatrix`: latency, FPR, precision, recall, F1 matrices of shape `(n_algorithms, n_variation_levels)`
