# DeltaPD

UHF Partial-Discharge inter-pulse interval (Δt) tracking framework with a thesis campaign mode.

## What It Does

DeltaPD now exposes two aligned but separate workflows.

1. **Legacy research mode** keeps the original four-phase Δt workflow:
   - Phase 1: stochastic wavelet optimisation
   - Phase 2: Δt extraction from detected pulses
   - Phase 3: Kalman / adaptive EWMA / CUSUM tracking
   - Phase 4: empirical Big-O and convergence diagnostics
2. **Thesis campaign mode** wraps the core without altering it and exports stable CSV artefacts for benchmark and gemelas datasets:
   - time-domain metrics (Vpp, peak, noise RMS, SNR, energy)
   - spectral metrics (band powers, shares, spectral centroid, high/low ratio)
   - Δt counts and mean Δt
   - detection curves Pd vs k·σ and summaries at 3σ, 5σ, and 7σ

## Quickstart

```bash
pip install -e .

# Legacy four-phase demo
python -m deltapd run-legacy --seed 42 -n 4096

# Thesis campaign mode
python -m deltapd run-thesis --config campaign/config_thesis.yaml

# Run tests
pytest
```

## Key Design Decision

The core Δt workflow remains intact. Thesis-specific metrics live in `src/deltapd/campaign/`, so the repository can evaluate sensor-level campaign data without distorting the original purpose of DeltaPD.

## Project Structure

```text
src/deltapd/
  __main__.py                # CLI with run-legacy / run-thesis
  pipeline.py                # Original four-phase pipeline
  loader.py                  # Polymorphic ingestion with optional amplitude preservation
  descriptors.py             # Pulse detection & Δt extraction
  trackers.py                # Kalman, Adaptive EWMA, CUSUM
  validation.py              # Big-O estimation and convergence matrix
  campaign/
    thesis_campaign.py       # Campaign orchestrator for benchmark/gemelas
    metrics_time.py          # Vpp, SNR, energy, noise RMS, z_peak
    metrics_spectral.py      # Band powers and centroid metrics
    detection_curves.py      # Pd vs k·σ tables
    aggregate.py             # Summary tables
campaign/
  config_thesis.yaml         # Example thesis configuration
```
