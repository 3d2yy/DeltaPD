import tempfile
from pathlib import Path

import pandas as pd

from deltapd.campaign.thesis_campaign import run_thesis_campaign
from deltapd.loader import load_empirical_signal


def test_loader_backward_and_trigger_modes():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "CH2.csv"
        path.write_text(
            "Date,19 NOV 2025\n"
            "Time,14:23:42\n"
            "0.0,0.0\n"
            "1e-9,1.0\n"
            "2e-9,0.5\n",
            encoding="utf-8",
        )
        signal, fs = load_empirical_signal(str(path))
        assert len(signal) == 3
        signal2, fs2, t = load_empirical_signal(
            str(path), include_trigger_time=True, preserve_amplitude=True
        )
        assert fs == fs2
        assert t > 0
        assert abs(signal2.mean()) < 1e-12
        assert max(signal2) == 0.5


def test_run_thesis_campaign_smoke():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "data"
        p1 = base / "Prueba 1 - Internas"
        p1.mkdir(parents=True)
        for ch, scale in [("CH2", 1.0), ("CH3", 2.0), ("CH4", 1.5)]:
            df = pd.DataFrame(
                {"t": [0.0, 1e-9, 2e-9, 3e-9], "v": [0.0, scale, 0.5 * scale, 0.0]}
            )
            df.to_csv(p1 / f"{ch}.csv", index=False)

        cfg = Path(td) / "config.yaml"
        cfg.write_text(
            """
base_dir: "{base_dir}"
output_dir: "outputs/thesis"
analysis_params:
  preserve_amplitude: true
  default_fs: 1.0e9
  noise_window_ns: 40
  delta_t_threshold_sigma: 1.0
  delta_t_detection_method: "threshold"
  detection_k_values: [3.0, 5.0, 7.0]
  bands_ghz:
    - [0.0, 0.5]
    - [0.5, 1.5]
    - [1.5, 2.4]
datasets:
  P1:
    folder: "Prueba 1 - Internas"
    label: "Prueba 1 - Internas"
    mode: "benchmark"
    channel_map:
      CH2: "Deepace"
      CH3: "AVA_propuesta"
      CH4: "Bioinspirada"
""".format(base_dir=str(base).replace("\\", "/")),
            encoding="utf-8",
        )
        outputs = run_thesis_campaign(cfg)
        assert not outputs["metrics"].empty
        assert "vpp" in outputs["metrics"].columns
        assert "pd_3sigma_percent" in outputs["detection_summary"].columns
