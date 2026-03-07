from __future__ import annotations

import numpy as np
import pandas as pd


def compute_detection_curves(metrics_df: pd.DataFrame, k_values: list[float]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if metrics_df.empty:
        return pd.DataFrame(columns=["dataset_key", "antenna", "k_sigma", "pd"])

    grouped = metrics_df.groupby(["dataset_key", "antenna"], dropna=False)
    for (dataset_key, antenna), grp in grouped:
        z = grp["z_peak"].dropna().to_numpy(dtype=float)
        if len(z) == 0:
            continue
        for k in k_values:
            pd_k = float(np.mean(z >= float(k)))
            rows.append({
                "dataset_key": dataset_key,
                "antenna": antenna,
                "k_sigma": float(k),
                "pd": pd_k,
                "pd_percent": 100.0 * pd_k,
            })
    return pd.DataFrame(rows)


def summarize_detection_curves(curves_df: pd.DataFrame, report_k: tuple[float, ...] = (3.0, 5.0, 7.0)) -> pd.DataFrame:
    if curves_df.empty:
        return pd.DataFrame(columns=["dataset_key", "antenna"])
    sub = curves_df[curves_df["k_sigma"].isin(report_k)].copy()
    wide = sub.pivot(index=["dataset_key", "antenna"], columns="k_sigma", values="pd_percent").reset_index()
    rename = {k: f"pd_{int(k)}sigma_percent" for k in report_k}
    wide = wide.rename(columns=rename)
    return wide
