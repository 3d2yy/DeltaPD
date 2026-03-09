# DeltaPD Visual Workbench

## 1. Decision

Yes: a visual environment is a good idea for this thesis.

Not as a replacement for the pipeline, but as a thin orchestration and review layer on top of it.

That is the correct architecture because it gives you:

- better presentation for thesis and papers;
- faster review of plots and tables;
- annotations on interesting regions;
- automatic PDF export;
- one workflow for single-test and multi-test studies;
- a path to support VNA files without mixing that logic into the core PD pipeline.

## 2. Non-negotiable rule

The visual layer must consume pipeline outputs and manifests.

It should not recalculate science in the browser or keep hidden logic that only exists in the UI.

In practice:

- the backend runs `material`, `descriptor`, `comparative`, or `state/alarm batch`;
- the UI reads `CSV`, `PNG`, `PDF`, and `JSON manifest` outputs;
- any parameter change writes a config and reruns the backend;
- every visual result must be reproducible from a saved config.

## 3. Recommended stack

- `Plotly Dash` for the web UI.
- Existing Python pipeline as the analysis backend.
- `reportlab` or backend-side PDF export for the final report.

Why Dash:

- it already matches the style of the example you sent;
- it is strong for scientific graphs and tabular dashboards;
- it is easier to keep inside one Python codebase;
- it is enough for thesis-grade interactivity without adding a heavy frontend stack.

## 4. Main modes

### A. PD single test

Input:

- one file or one folder;
- choose channel;
- choose analysis mode: `material` or `state/alarm`.

Output:

- raw waveform and detections;
- delta t plots;
- blind PRPD;
- rolling stats;
- candidate transition windows;
- descriptor ranking;
- tables;
- PDF report.

### B. PD comparative

Input:

- multiple tests selected together.

Behavior:

- if the user selects more than one compatible test, the app switches to comparative mode automatically.

Output:

- descriptor comparison across tests;
- type-separation plots;
- dataset heatmaps;
- summary tables;
- comparative PDF.

### C. VNA single test

Input:

- one VNA file.

Behavior:

- auto-detect columns;
- identify whether the file contains `S11`, return loss, or noise-like spectra;
- if `S11` is available, compute `VSWR`;
- generate the key graphs automatically.

Output:

- `S11` vs frequency;
- `VSWR` vs frequency;
- minima, bandwidth and resonance markers;
- optional comparison against another file;
- PDF report.

### D. VNA comparative

Input:

- two or more VNA files.

Output:

- overlay plots;
- delta curves between runs;
- resonance shift table;
- bandwidth comparison;
- PDF report.

## 5. Automatic behavior

The app should ask for files or folders and infer the mode:

- if one PD test is selected: run single-test mode;
- if several PD tests are selected: run comparative PD mode;
- if one VNA file is selected: run VNA single mode;
- if several VNA files are selected: run VNA comparative mode.

The user should still be able to override the detected mode manually.

## 6. Configurable controls that are worth exposing

- pulse detection threshold `k_sigma`;
- refractory time;
- wavelet on/off;
- rolling window size;
- max valid `delta t`;
- descriptor bank selection;
- task type: `state`, `alarm`, `type`, `variant`, `dataset`;
- PDF title and report notes.

Do not expose every internal constant on day one. Start with the parameters above.

## 7. Visual sections

The dashboard should have these tabs:

1. `Inputs`
2. `Run setup`
3. `Plots`
4. `Descriptors`
5. `Transitions`
6. `Tables`
7. `Report`

## 8. Styling direction

Use a more professional scientific look than the demo app:

- white or warm-light background;
- deep blue, copper, and neutral gray palette;
- readable serif or humanist headings;
- restrained animations only;
- no oversaturated colors;
- no "BI dashboard" look.

## 9. What is already ready in the repo

These backend outputs now exist and are suitable for a future UI:

- comparative CH3 study outputs;
- per-test `state/alarm` batch outputs;
- per-case PDFs;
- master summaries;
- machine-readable manifests.

That means the UI can be started without redesigning the science layer first.

## 10. Recommended build order

1. Build a minimal `Dash` shell that reads existing manifests and shows existing plots.
2. Add rerun buttons that execute saved configs.
3. Add report assembly and note-taking fields.
4. Add VNA ingestion mode.
5. Add comparison presets and styling polish.

## 11. Final recommendation

Yes, build it.

But build it as a controlled thesis workbench, not as a separate analytical system.

If we keep that boundary, it will help the thesis a lot and it will not weaken the methodological defense.
