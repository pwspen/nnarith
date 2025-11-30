from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def normalize_reducer_name(name: str) -> str:
    try:
        key = name.lower()
    except AttributeError as exc:  # pragma: no cover - defensive
        raise TypeError("Reducer must be a string identifier.") from exc
    aliases = {
        "average": "average",
        "avg": "average",
        "mean": "average",
        "median": "median",
        "max": "max",
        "maximum": "max",
        "min": "min",
        "minimum": "min",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown reducer '{name}'. Available reducers: average, median, max, min."
        )
    return aliases[key]


def aggregate_block(block: np.ndarray, mode: str) -> float:
    arr = np.asarray(block, dtype=float)
    if arr.size == 0:
        return float("nan")
    mask = ~np.isnan(arr)
    if not mask.any():
        return float("nan")
    data = arr[mask]
    if mode == "average":
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(np.mean(data))
    if mode == "median":
        with np.errstate(invalid="ignore"):
            return float(np.median(data))
    if mode == "max":
        return float(np.max(data))
    return float(np.min(data))


def maybe_downsample_matrix(arr: np.ndarray, max_dim: int, reducer: str) -> Tuple[np.ndarray, bool]:
    if max_dim < 1:
        raise ValueError("max_display_size must be at least 1.")
    if arr.ndim != 2:
        raise ValueError("Reducer expects a 2D array.")
    m, n = arr.shape
    if m == 0 or n == 0:
        return arr, False
    target_m = min(m, max_dim)
    target_n = min(n, max_dim)
    if target_m == m and target_n == n:
        return arr, False
    row_bins = np.array_split(np.arange(m), target_m)
    col_bins = np.array_split(np.arange(n), target_n)
    reduced = np.empty((target_m, target_n), dtype=float)
    for i, rows in enumerate(row_bins):
        for j, cols in enumerate(col_bins):
            block = arr[np.ix_(rows, cols)]
            reduced[i, j] = aggregate_block(block, reducer)
    return reduced, True


def to_ndarray(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach() if value.requires_grad else value
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            return tensor.numpy()
    except ImportError:
        pass

    try:
        import tensorflow as tf  # type: ignore

        if isinstance(value, tf.Tensor):
            return value.numpy()
    except ImportError:
        pass

    try:
        return np.asarray(value)
    except Exception as exc:
        raise TypeError(
            "Unable to convert value to a NumPy array; ensure the input is array-like."
        ) from exc


def format_number(x: float, precision: Optional[int], is_int: bool) -> str:
    if not np.isfinite(x):
        if np.isnan(x):
            return "NaN"
        return "+Inf" if x > 0 else "-Inf"
    if is_int:
        return str(int(round(x)))
    if precision is not None:
        return f"{x:.{precision}f}"
    ax = abs(x)
    if (ax != 0 and ax < 1e-3) or ax >= 1e4:
        return f"{x:.2e}"
    return f"{x:.3f}"


__all__ = [
    "aggregate_block",
    "format_number",
    "maybe_downsample_matrix",
    "normalize_reducer_name",
    "to_ndarray",
]
