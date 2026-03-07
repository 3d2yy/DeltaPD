# DeltaPD thesis refactor audit

## Objective

Keep the original DeltaPD purpose intact as a \(\Delta t\)-centric framework while adding a separate, thesis-aligned campaign mode for benchmark and gemelas datasets.

## Core corrections applied

1. **CLI repaired**
   - `src/deltapd/__main__.py` no longer points to a non-existent interface.
   - Added explicit subcommands:
     - `run-legacy`
     - `run-thesis --config campaign/config_thesis.yaml`

2. **Loader repaired and extended**
   - `load_empirical_signal()` now supports:
     - single-column CSV waveform files,
     - optional amplitude preservation,
     - optional trigger time return.
   - Default behavior remains normalized output for backward compatibility.
   - Thesis mode can now use `preserve_amplitude=True` to keep Vpp, SNR, and energy physically meaningful.

3. **Legacy pipeline preserved**
   - Added a real `main()` entrypoint in `pipeline.py` for the original four-phase workflow.
   - Kept the original research purpose centered on pulse detection, \(\Delta t\) extraction, tracking, and validation.

4. **New thesis campaign layer added**
   - Added `src/deltapd/campaign/` with the following modules:
     - `config.py`
     - `metrics_time.py`
     - `metrics_spectral.py`
     - `detection_curves.py`
     - `aggregate.py`
     - `thesis_campaign.py`
   - This layer does **not** alter the DeltaPD core.
   - It wraps the core and exports stable thesis artefacts as CSV tables.

5. **Configuration externalized**
   - Added `campaign/config_thesis.yaml` example.
   - Benchmark and gemelas are now explicitly separated through dataset-specific `channel_map` blocks.

## Thesis outputs now supported

The `run-thesis` command exports:

- `thesis_metrics.csv`
- `thesis_summary_by_dataset_antenna.csv`
- `thesis_detection_curves.csv`
- `thesis_detection_summary_3_5_7sigma.csv`

These files are intended to be the stable source of truth for Chapter IV tables and figure scripts.

## Design boundary respected

The refactor intentionally **does not** turn DeltaPD into a Chapter IV figure factory.
The design keeps two distinct layers:

- **DeltaPD core**: evaluates signal evolution through pulse detection, \(\Delta t\), tracking, and validation.
- **Thesis campaign mode**: evaluates sensor-level campaign metrics without modifying the core scientific meaning of DeltaPD.

## Validation status

- Test suite result after refactor: **26 passed**

## Remaining work (optional, not required for conceptual alignment)

1. Add a `plot_cap4.py` module if final Chapter IV figures should be emitted directly from the repo.
2. Add richer campaign loaders if oscilloscope exports are heterogeneous across campaigns.
3. Add regression tests with real campaign files once a frozen sample dataset is available.
