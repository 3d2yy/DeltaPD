from __future__ import annotations

import pandas as pd


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    numeric_cols = metrics_df.select_dtypes(include=["number"]).columns.tolist()
    group_cols = ["dataset_key", "dataset_label", "mode", "antenna"]
    keep_num = [c for c in numeric_cols if c not in {"trigger_time_epoch"}]

    agg = {}
    for col in keep_num:
        agg[col] = ["mean", "median"]

    out = metrics_df.groupby(group_cols, dropna=False).agg(agg)
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    out = out.reset_index()
    return out
