import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from deltapd.vna import analyze_vna_selection


def test_analyze_vna_selection_csv_s11_generates_vswr_outputs():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        csv_path = root / "antena.csv"
        pd.DataFrame(
            {
                "frequency_ghz": [0.8, 1.0, 1.2],
                "s11_db": [-8.0, -18.0, -10.5],
            }
        ).to_csv(csv_path, index=False)

        outputs = analyze_vna_selection([csv_path], root / "out")
        summary = pd.read_csv(outputs["summary_csv"])

        assert summary.loc[0, "mode"] == "s11"
        assert summary.loc[0, "min_s11_db"] <= -18.0
        assert (root / "out" / "antena" / "vna_overview.png").exists()
        assert (root / "out" / "vna_manifest.json").exists()
        assert (root / "out" / "vna_report.pdf").exists()


def test_analyze_vna_selection_touchstone_comparative_overlay():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        file_a = root / "a.s1p"
        file_b = root / "b.s1p"
        file_a.write_text(
            "\n".join(
                [
                    "# GHZ S DB R 50",
                    "0.8 -10 0",
                    "1.0 -20 0",
                    "1.2 -12 0",
                ]
            ),
            encoding="utf-8",
        )
        file_b.write_text(
            "\n".join(
                [
                    "# GHZ S DB R 50",
                    "0.8 -9 0",
                    "1.0 -15 0",
                    "1.2 -11 0",
                ]
            ),
            encoding="utf-8",
        )

        outputs = analyze_vna_selection([file_a, file_b], root / "out")
        manifest_text = (root / "out" / "vna_manifest.json").read_text(encoding="utf-8")

        assert outputs["summary_df"].shape[0] == 2
        assert (root / "out" / "vna_comparative_overlay.png").exists()
        assert (root / "out" / "vna_report.pdf").exists()
        assert "comparative" in manifest_text
