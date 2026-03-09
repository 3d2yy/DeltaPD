"""
Polymorphic empirical signal loader for UHF-PD traces.

The loader supports CSV, MATLAB (.mat), and HDF5 (.h5/.hdf5) inputs and
returns a one-dimensional voltage trace plus sampling frequency. For thesis
campaign work, amplitude preservation can be enabled so that Vpp, energy, and
SNR remain physically comparable across sensors.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import loadmat

Signal = NDArray[np.floating[Any]]

KEYSIGHT_SEGMENT_PREFIX = "Time Since Seg 1"
CSV_DELIMITER_CANDIDATES = [",", ";", "\t", "|"]
TIME_COLUMN_TOKENS = ("time", "tiempo", "timestamp", "seconds", "sec", "t", "x")
SIGNAL_COLUMN_TOKENS = (
    "voltage",
    "volt",
    "signal",
    "amplitude",
    "trace",
    "waveform",
    "channel",
    "ch",
    "v",
    "y",
)
INDEX_COLUMN_TOKENS = ("index", "sample", "samples", "n", "idx")


def _normalize_amplitude(signal: Signal) -> Signal:
    """Center and normalize a signal to the range [-1, 1]."""
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    max_val = np.max(np.abs(signal)) if signal.size else 0.0
    if max_val == 0:
        return signal
    return signal / max_val


def _center_only(signal: Signal) -> Signal:
    """Remove DC offset while preserving absolute amplitude scale."""
    signal = np.asarray(signal, dtype=np.float64)
    return signal - np.mean(signal)


def _finalize_signal(raw_signal: Signal, preserve_amplitude: bool) -> Signal:
    raw_signal = np.asarray(raw_signal, dtype=np.float64)
    raw_signal = raw_signal[~np.isnan(raw_signal)]
    if preserve_amplitude:
        return _center_only(raw_signal)
    return _normalize_amplitude(raw_signal)


def _infer_trigger_time(date_str: str, time_str: str) -> float:
    if not date_str or not time_str:
        return 0.0
    try:
        dt_obj = datetime.strptime(f"{date_str} {time_str}", "%d %b %Y %H:%M:%S")
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        return float(dt_obj.timestamp())
    except ValueError:
        return 0.0


def _normalize_csv_key(value: str) -> str:
    return str(value).strip().replace(":", "")


def _normalize_column_label(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _coerce_float_token(value: str) -> float:
    token = str(value).strip().strip('"').replace(" ", "")
    if not token:
        raise ValueError("Empty token")

    token = token.replace("−", "-")
    if "," in token:
        if "." not in token:
            token = token.replace(",", ".")
        elif token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    return float(token)


def _is_float_token(value: str) -> bool:
    try:
        _coerce_float_token(value)
    except ValueError:
        return False
    return True


def _extract_csv_metadata(rows: list[list[str]], max_rows: int = 40) -> dict[str, list[str]]:
    metadata: dict[str, list[str]] = {}
    for row in rows[:max_rows]:
        if not row:
            continue
        key = _normalize_csv_key(row[0])
        if not key or _is_float_token(key):
            continue
        metadata[key] = [str(item).strip() for item in row[1:]]
    return metadata


def _find_numeric_data_start(rows: list[list[str]]) -> int:
    for idx, row in enumerate(rows):
        cleaned = [str(item).strip() for item in row if str(item).strip()]
        if len(cleaned) >= 1 and all(_is_float_token(token) for token in cleaned[: min(4, len(cleaned))]):
            return idx
    return len(rows)


def _to_float_array(values: list[str]) -> NDArray[np.float64]:
    cleaned = [value for value in values if str(value).strip()]
    if not cleaned:
        return np.array([], dtype=np.float64)
    return np.array([_coerce_float_token(value) for value in cleaned], dtype=np.float64)


def _metadata_row(metadata: dict[str, list[str]], key: str) -> list[str]:
    if key in metadata:
        return metadata[key]
    for candidate, values in metadata.items():
        if candidate.startswith(key):
            return values
    return []


def _looks_like_keysight_segmented(metadata: dict[str, list[str]]) -> bool:
    return bool(
        _metadata_row(metadata, "XInc")
        and _metadata_row(metadata, "Points")
        and _metadata_row(metadata, KEYSIGHT_SEGMENT_PREFIX)
    )


def _infer_csv_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(CSV_DELIMITER_CANDIDATES))
        return str(dialect.delimiter)
    except csv.Error:
        counts = {delimiter: sample.count(delimiter) for delimiter in CSV_DELIMITER_CANDIDATES}
        return max(counts, key=counts.get)


def _read_csv_rows(file_path: str) -> list[list[str]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        delimiter = _infer_csv_delimiter(sample)
        reader = csv.reader(f, delimiter=delimiter)
        return [list(row) for row in reader]


def _infer_fs_from_metadata(metadata: dict[str, list[str]], default_fs: float) -> tuple[float, str]:
    fs = float(default_fs)
    sample_rate_keys = [
        "SampleRate",
        "Sample Rate",
        "SamplingRate",
        "Sampling Rate",
        "Fs",
        "F_s",
        "Frecuencia",
    ]
    for key in sample_rate_keys:
        values = _metadata_row(metadata, key)
        if not values:
            continue
        numeric = _to_float_array(values)
        numeric = numeric[np.isfinite(numeric) & (numeric > 0)]
        if numeric.size:
            return float(numeric[0]), "metadata_sample_rate"

    xinc = _to_float_array(_metadata_row(metadata, "XInc"))
    xinc = xinc[np.isfinite(xinc) & (xinc > 0)]
    if xinc.size:
        return float(1.0 / np.median(xinc)), "metadata_xinc"
    return fs, "default_fs"


def _last_nonempty_row(rows: list[list[str]], stop_idx: int) -> list[str]:
    if not rows:
        return []
    for idx in range(max(stop_idx - 1, 0), -1, -1):
        cleaned = [str(item).strip() for item in rows[idx] if str(item).strip()]
        if cleaned:
            return cleaned
    return []


def _is_time_like_column(values: NDArray[np.float64]) -> bool:
    if len(values) < 3:
        return False
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        return False
    positive = diffs > 0
    return bool(np.mean(positive) >= 0.95)


def _header_token_matches(label: str, tokens: tuple[str, ...]) -> bool:
    normalized = _normalize_column_label(label)
    for token in tokens:
        if len(token) <= 1:
            if normalized == token:
                return True
            continue
        if normalized == token or normalized.startswith(token) or normalized.endswith(token):
            return True
    return False


def _select_generic_columns(
    mat: NDArray[np.float64],
    header_row: list[str],
) -> tuple[int | None, int]:
    n_cols = int(mat.shape[1])
    headers = [header_row[idx] if idx < len(header_row) else "" for idx in range(n_cols)]

    time_idx = next((idx for idx, label in enumerate(headers) if _header_token_matches(label, TIME_COLUMN_TOKENS)), None)
    signal_idx = next((idx for idx, label in enumerate(headers) if _header_token_matches(label, SIGNAL_COLUMN_TOKENS) and idx != time_idx), None)
    index_idx = next((idx for idx, label in enumerate(headers) if _header_token_matches(label, INDEX_COLUMN_TOKENS)), None)

    if time_idx is None:
        for idx in range(n_cols):
            if index_idx is not None and idx == index_idx:
                continue
            if _is_time_like_column(mat[:, idx]):
                time_idx = idx
                break

    if signal_idx is None:
        candidate_cols = [idx for idx in range(n_cols) if idx != time_idx and idx != index_idx]
        if not candidate_cols:
            candidate_cols = [idx for idx in range(n_cols) if idx != time_idx]
        signal_idx = max(candidate_cols, key=lambda idx: float(np.nanstd(mat[:, idx]))) if candidate_cols else 0

    if time_idx == signal_idx:
        time_idx = None
    return time_idx, int(signal_idx)


def _load_keysight_segmented_csv(
    rows: list[list[str]],
    metadata: dict[str, list[str]],
    default_fs: float,
    *,
    delimiter: str,
) -> tuple[Signal, float, float, NDArray[np.float64] | None, dict[str, Any]]:
    data_start = _find_numeric_data_start(rows)
    numeric_rows: list[list[float]] = []
    for row in rows[data_start:]:
        cleaned = [str(item).strip() for item in row if str(item).strip()]
        if not cleaned:
            continue
        try:
            numeric_rows.append([float(item) for item in cleaned])
        except ValueError:
            continue

    if not numeric_rows:
        raise ValueError("No numeric segmented waveform data found in CSV.")

    mat = np.array(numeric_rows, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] < 2:
        raise ValueError("Segmented CSV structure is invalid.")

    time_cols = mat[:, 0::2]
    signal_cols = mat[:, 1::2]

    xinc = _to_float_array(_metadata_row(metadata, "XInc"))
    xorg = _to_float_array(_metadata_row(metadata, "XOrg"))
    segment_offsets = _to_float_array(_metadata_row(metadata, KEYSIGHT_SEGMENT_PREFIX))

    fs, fs_source = _infer_fs_from_metadata(metadata, default_fs)

    times = time_cols
    segment_starts = time_cols[0, :] if time_cols.size else np.array([], dtype=np.float64)
    has_absolute_segment_times = bool(segment_starts.size >= 2 and np.all(np.diff(segment_starts) > 0))
    used_segment_offsets = False

    if not has_absolute_segment_times and xinc.size and xorg.size and segment_offsets.size:
        n_rows, n_segments = time_cols.shape
        if xinc.size >= n_segments and xorg.size >= n_segments and segment_offsets.size >= n_segments:
            sample_idx = np.arange(n_rows, dtype=np.float64)[:, None]
            times = segment_offsets[:n_segments][None, :] + xorg[:n_segments][None, :] + sample_idx * xinc[:n_segments][None, :]
            used_segment_offsets = True

    flat_times = times.flatten(order="F")
    raw_signal = signal_cols.flatten(order="F")

    date_vals = _metadata_row(metadata, "Date")
    time_vals = _metadata_row(metadata, "Time")
    date_str = date_vals[0] if date_vals else ""
    time_str = time_vals[0] if time_vals else ""
    t_trig = _infer_trigger_time(date_str, time_str)
    diagnostics = {
        "file_type": "csv",
        "loader_mode": "keysight_segmented_csv",
        "delimiter": delimiter,
        "fs_hz": float(fs),
        "fs_source": fs_source,
        "sample_count": int(len(raw_signal)),
        "numeric_row_count": int(mat.shape[0]),
        "column_count": int(mat.shape[1]),
        "time_column_label": "interleaved_time_columns",
        "signal_column_label": "interleaved_signal_columns",
        "has_absolute_times": True,
        "used_segment_offsets": bool(used_segment_offsets),
        "metadata_keys_count": int(len(metadata)),
    }
    return raw_signal, fs, t_trig, flat_times, diagnostics


def _load_generic_csv_signal(
    rows: list[list[str]],
    metadata: dict[str, list[str]],
    default_fs: float,
    *,
    delimiter: str,
) -> tuple[Signal, float, float, NDArray[np.float64] | None, dict[str, Any]]:
    data_start = _find_numeric_data_start(rows)
    header_row = _last_nonempty_row(rows, data_start)
    numeric_rows: list[list[float]] = []
    expected_cols: int | None = None

    for row in rows[data_start:]:
        cleaned = [str(item).strip() for item in row if str(item).strip()]
        if not cleaned:
            continue
        try:
            floats = [_coerce_float_token(item) for item in cleaned]
        except ValueError:
            continue
        if expected_cols is None:
            expected_cols = len(floats)
        if len(floats) != expected_cols:
            continue
        numeric_rows.append(floats)

    if not numeric_rows:
        raise ValueError("No numeric waveform data found in CSV.")

    mat = np.array(numeric_rows, dtype=np.float64)
    times: NDArray[np.float64] | None = None
    if mat.ndim != 2 or mat.shape[1] == 0:
        raise ValueError("CSV numeric data is malformed.")

    time_label = ""
    signal_label = ""
    fs, fs_source = _infer_fs_from_metadata(metadata, default_fs)
    if mat.shape[1] == 1:
        raw_signal = mat[:, 0]
        signal_label = header_row[0] if header_row else "single_numeric_column"
    elif mat.shape[1] % 2 == 0 and mat.shape[1] > 2 and all(
        _is_time_like_column(mat[:, idx]) for idx in range(0, mat.shape[1], 2)
    ):
        times = mat[:, 0::2].flatten(order="F")
        raw_signal = mat[:, 1::2].flatten(order="F")
        time_label = "interleaved_time_columns"
        signal_label = "interleaved_signal_columns"
    else:
        time_idx, signal_idx = _select_generic_columns(mat, header_row)
        raw_signal = mat[:, signal_idx]
        signal_label = header_row[signal_idx] if signal_idx < len(header_row) else f"column_{signal_idx}"
        if time_idx is not None:
            times = mat[:, time_idx]
            time_label = header_row[time_idx] if time_idx < len(header_row) else f"column_{time_idx}"
    if times is not None and len(times) >= 2:
        diffs = np.diff(times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            dt_est = float(np.median(diffs[: min(1000, diffs.size)]))
        else:
            dt_est = 0.0
        if dt_est > 0:
            fs = float(1.0 / dt_est)
            fs_source = "time_axis"

    date_vals = _metadata_row(metadata, "Date")
    time_vals = _metadata_row(metadata, "Time")
    date_str = date_vals[0] if date_vals else ""
    time_str = time_vals[0] if time_vals else ""
    t_trig = _infer_trigger_time(date_str, time_str)
    diagnostics = {
        "file_type": "csv",
        "loader_mode": "generic_csv",
        "delimiter": delimiter,
        "fs_hz": float(fs),
        "fs_source": fs_source,
        "sample_count": int(len(raw_signal)),
        "numeric_row_count": int(mat.shape[0]),
        "column_count": int(mat.shape[1]),
        "time_column_label": str(time_label),
        "signal_column_label": str(signal_label),
        "has_absolute_times": bool(times is not None),
        "used_segment_offsets": False,
        "metadata_keys_count": int(len(metadata)),
    }
    return raw_signal, fs, t_trig, times, diagnostics


def _load_csv_signal(file_path: str, default_fs: float) -> tuple[Signal, float, float, NDArray[np.float64] | None, dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        delimiter = _infer_csv_delimiter(f.read(8192))
    rows = _read_csv_rows(file_path)
    metadata = _extract_csv_metadata(rows)
    if _looks_like_keysight_segmented(metadata):
        return _load_keysight_segmented_csv(rows, metadata, default_fs, delimiter=delimiter)
    return _load_generic_csv_signal(rows, metadata, default_fs, delimiter=delimiter)


def _load_mat_signal(file_path: str, default_fs: float) -> tuple[Signal, float, float, NDArray[np.float64] | None, dict[str, Any]]:
    mat_data = loadmat(file_path)
    keys = [k for k in mat_data.keys() if not k.startswith("__")]
    raw_signal: Signal = np.array([], dtype=np.float64)
    fs = float(default_fs)

    signal_keys = ["x3", "data", "signal", "volts", "voltage", "y"]
    for sk in signal_keys:
        if sk in mat_data:
            val = np.squeeze(mat_data[sk])
            if val.ndim == 1 and len(val) > 100:
                raw_signal = val.astype(np.float64)
                break

    if len(raw_signal) == 0:
        best_key = ""
        max_len = 0
        for k in keys:
            val = np.squeeze(mat_data[k])
            if val.ndim == 1 and len(val) > max_len:
                max_len = len(val)
                best_key = k
        if best_key:
            raw_signal = np.squeeze(mat_data[best_key]).astype(np.float64)

    for k in keys:
        if k.lower() in ["fs", "samplerate", "f_s", "frecuencia"]:
            fs = float(np.squeeze(mat_data[k]))
            break

    if len(raw_signal) == 0:
        raise ValueError("No valid 1-D waveform found in .mat file.")

    diagnostics = {
        "file_type": "mat",
        "loader_mode": "matlab_array",
        "delimiter": "",
        "fs_hz": float(fs),
        "fs_source": "mat_metadata" if fs != float(default_fs) else "default_fs",
        "sample_count": int(len(raw_signal)),
        "numeric_row_count": int(len(raw_signal)),
        "column_count": 1,
        "time_column_label": "",
        "signal_column_label": "mat_signal",
        "has_absolute_times": False,
        "used_segment_offsets": False,
        "metadata_keys_count": int(len(keys)),
    }
    return raw_signal, fs, 0.0, None, diagnostics


def _load_hdf5_signal(file_path: str, default_fs: float) -> tuple[Signal, float, float, NDArray[np.float64] | None, dict[str, Any]]:
    raw_signal: Signal = np.array([], dtype=np.float64)
    fs = float(default_fs)
    with h5py.File(file_path, "r") as f:
        def _find_dataset(name: str, obj: Any) -> None:
            nonlocal raw_signal, fs
            if isinstance(obj, h5py.Dataset):
                if obj.ndim == 1 and obj.size > 100 and len(raw_signal) == 0:
                    raw_signal = obj[:].astype(np.float64)
                elif obj.size == 1 and ("fs" in name.lower() or "rate" in name.lower()):
                    fs = float(obj[...])

        f.visititems(_find_dataset)
        if "fs" in f.attrs:
            fs = float(f.attrs["fs"])

    if len(raw_signal) == 0:
        raise ValueError("HDF5 structure incompatible: no waveform found.")

    diagnostics = {
        "file_type": "hdf5",
        "loader_mode": "hdf5_dataset",
        "delimiter": "",
        "fs_hz": float(fs),
        "fs_source": "hdf5_metadata" if fs != float(default_fs) else "default_fs",
        "sample_count": int(len(raw_signal)),
        "numeric_row_count": int(len(raw_signal)),
        "column_count": 1,
        "time_column_label": "",
        "signal_column_label": "hdf5_signal",
        "has_absolute_times": False,
        "used_segment_offsets": False,
        "metadata_keys_count": 0,
    }
    return raw_signal, fs, 0.0, None, diagnostics


def load_empirical_signal(
    file_path: str,
    default_fs: float = 1e9,
    *,
    preserve_amplitude: bool = False,
    include_trigger_time: bool = False,
    include_absolute_times: bool = False,
    include_diagnostics: bool = False,
) -> tuple[Signal, float] | tuple[Signal, float, float] | tuple[Signal, float, float, NDArray[np.float64] | None] | tuple[Signal, float, NDArray[np.float64] | None] | tuple[Any, ...]:
    """Load and homogenize an empirical UHF-PD waveform.

    Parameters
    ----------
    file_path:
        File path to .csv, .mat, .h5, or .hdf5 waveform data.
    default_fs:
        Fallback sampling frequency in Hz.
    preserve_amplitude:
        If ``True``, only remove DC offset and preserve the original amplitude
        scale. If ``False`` (default), return a zero-mean normalized waveform.
    include_trigger_time:
        If ``True``, also return the trigger timestamp as a Unix epoch float.
    include_absolute_times:
        If ``True``, also returns the precise absolute timestamp array for each sample
        (crucial for segmented captures like Rigol CSVs where time jumps).

    Returns
    -------
    Depending on flags: signal, fs, [t_trig], [times], [diagnostics]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".csv":
        raw_signal, fs, t_trig, times, diagnostics = _load_csv_signal(file_path, default_fs)
    elif ext == ".mat":
        raw_signal, fs, t_trig, times, diagnostics = _load_mat_signal(file_path, default_fs)
    elif ext in {".h5", ".hdf5"}:
        raw_signal, fs, t_trig, times, diagnostics = _load_hdf5_signal(file_path, default_fs)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Utilice .csv, .mat o .h5")

    signal = _finalize_signal(raw_signal, preserve_amplitude=preserve_amplitude)
    diagnostics["preserve_amplitude"] = bool(preserve_amplitude)
    diagnostics["final_sample_count"] = int(len(signal))
    diagnostics["nan_filtered_count"] = int(len(np.asarray(raw_signal)) - len(signal))

    result: tuple[Any, ...]
    if include_trigger_time and include_absolute_times:
        result = (signal, fs, t_trig, times)
    elif include_trigger_time:
        result = (signal, fs, t_trig)
    elif include_absolute_times:
        result = (signal, fs, times)
    else:
        result = (signal, fs)

    if include_diagnostics:
        result = result + (diagnostics,)
    return result
