# DeltaPD Workbench Usage

## Launch

Install the package once in editable mode, including the UI extras:

```bash
pip install -e .[ui]
```

Run:

```bash
python -m deltapd run-workbench
```

Optional:

```bash
python -m deltapd run-workbench --host 127.0.0.1 --port 8050
```

You can also point the workbench to a specific thesis base folder or to previously generated outputs:

```bash
python -m deltapd run-workbench \
  --pd-base-dir "E:/Carpeta definitiva de Tesis/programas" \
  --state-alarm-root "e:/DeltaDP/outputs/state_alarm_ch3" \
  --comparative-root "e:/DeltaDP/outputs/comparative_ch3"
```

Then open:

`http://127.0.0.1:8050`

## What it does

- discovers available thesis PD datasets under the configured base directory;
- launches a `state/alarm` study automatically when you select one PD test;
- launches a comparative PD study automatically when you select several PD tests;
- accepts one or more VNA files or folders and decides between single or comparative analysis;
- reads the generated `state_alarm`, `comparative`, and `VNA` manifests;
- shows summary cards, score plots, recurrent descriptors, generated tables, images, artifact links, and PDFs.

## Inputs tab workflow

### PD runner

- Select one test if you want an individual `state/alarm` study.
- Select several tests if you want a comparative descriptor study.
- Adjust `k sigma` and toggle wavelet preprocessing before launching.
- The workbench writes a temporary config under `outputs/workbench_runtime/...` and runs the appropriate pipeline automatically.

### VNA runner

- Paste one file path per line, or a folder path.
- Supported file types include `.s1p`, `.s2p`, `.csv`, `.txt`, and `.dat`.
- If the file contains `S11`, the workbench generates `S11 + VSWR`.
- If the file does not expose `S11` but has a frequency-response column, the workbench falls back to generic response mode.
- If several compatible files are provided, it generates a comparative overlay.
- The VNA pipeline also exports `CSV + PNG + Markdown + JSON + PDF`.

## Tabs

### Overview

- Batch summary for the currently active `state/alarm` output.
- Score plots and recurrent descriptor counts.

### Case Review

- Detailed per-test metrics, narrative, figures, artifacts, and PDF.

### Comparative

- Comparative study narrative, feature recommendations, figures, and PDF.

### VNA

- Parsed VNA tables, narrative summary, generated plots, comparative overlay when available, and PDF report.

## Runtime outputs

The workbench keeps generated runs under:

- `outputs/workbench_runtime/state_alarm_selection_*`
- `outputs/workbench_runtime/comparative_selection_*`
- `outputs/workbench_runtime/vna_selection_*`

## Main data sources

- `outputs/state_alarm_ch3/state_alarm_batch_manifest.json`
- `outputs/state_alarm_ch3/state_alarm_batch_summary.csv`
- `outputs/comparative_ch3/comparative_summary.md`
- `outputs/comparative_ch3/study_recommendations.json`
- generated PDFs, PNGs, CSVs and Markdown reports under those output folders
- runtime outputs under `outputs/workbench_runtime/...`

## Current boundaries

- The workbench orchestrates studies and reads their outputs, but it does not yet edit thesis notes or annotations inside the UI.
- The current auto-discovery for PD studies is centered on thesis datasets with `CH3.csv`.
