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


def _load_csv_signal(file_path: str, default_fs: float) -> tuple[Signal, float, float, NDArray[np.float64] | None]:
    volts1d: list[float] = []
    matrix = []
    date_str = ""
    time_str = ""
    is_interleaved = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            key = str(row[0]).strip().replace(":", "")
            if key == "Date" and len(row) > 1:
                date_str = str(row[1]).strip()
                continue
            if key == "Time" and len(row) > 1:
                time_str = str(row[1]).strip()
                continue

            cleaned = [x.strip() for x in row if str(x).strip()]
            if not cleaned:
                continue

            # Pure single-column waveform file
            if len(cleaned) == 1:
                try:
                    volts1d.append(float(cleaned[0]))
                except ValueError:
                    continue
                continue

            try:
                floats = [float(x) for x in cleaned]
                matrix.append(floats)
                if len(floats) >= 2:
                    is_interleaved = True
            except ValueError:
                continue

    if len(volts1d) == 0 and len(matrix) == 0:
        raise ValueError("No numeric waveform data found in CSV.")

    times: NDArray[np.float64] | None = None
    if is_interleaved and len(matrix) > 0:
        mat = np.array(matrix, dtype=np.float64)
        if mat.shape[1] == 2:
            times = mat[:, 0]
            raw_signal = mat[:, 1]
        elif mat.shape[1] % 2 == 0:
            # Flatten column-major (Fortran order) to keep segments intact
            times = mat[:, 0::2].flatten(order='F')
            raw_signal = mat[:, 1::2].flatten(order='F')
        else:
            times = mat[:, 0]
            raw_signal = mat[:, 1]
    else:
        raw_signal = np.array(volts1d, dtype=np.float64)

    fs = float(default_fs)
    if times is not None and len(times) >= 2:
        # Determine Fs from the median of first 1000 deltas
        dt_est = np.median(np.diff(times[:1000]))
        if dt_est > 0:
            fs = float(1.0 / dt_est)
            
    t_trig = _infer_trigger_time(date_str, time_str)
    return raw_signal, fs, t_trig, times


def _load_mat_signal(file_path: str, default_fs: float) -> tuple[Signal, float, float, NDArray[np.float64] | None]:
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

    return raw_signal, fs, 0.0, None


def _load_hdf5_signal(file_path: str, default_fs: float) -> tuple[Signal, float, float, NDArray[np.float64] | None]:
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

    return raw_signal, fs, 0.0, None


def load_empirical_signal(
    file_path: str,
    default_fs: float = 1e9,
    *,
    preserve_amplitude: bool = False,
    include_trigger_time: bool = False,
    include_absolute_times: bool = False,
) -> tuple[Signal, float] | tuple[Signal, float, float] | tuple[Signal, float, float, NDArray[np.float64] | None] | tuple[Signal, float, NDArray[np.float64] | None]:
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
    Depending on falgs: signal, fs, [t_trig], [times]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".csv":
        raw_signal, fs, t_trig, times = _load_csv_signal(file_path, default_fs)
    elif ext == ".mat":
        raw_signal, fs, t_trig, times = _load_mat_signal(file_path, default_fs)
    elif ext in {".h5", ".hdf5"}:
        raw_signal, fs, t_trig, times = _load_hdf5_signal(file_path, default_fs)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Utilice .csv, .mat o .h5")

    signal = _finalize_signal(raw_signal, preserve_amplitude=preserve_amplitude)
    
    if include_trigger_time and include_absolute_times:
        return signal, fs, t_trig, times
    if include_trigger_time:
        return signal, fs, t_trig
    if include_absolute_times:
        return signal, fs, times
        
    return signal, fs
