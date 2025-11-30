import shutil
from typing import Any, Optional, Tuple
import numpy as np

# Minimal palettes (anchor colors) and linear interpolation
# Colors are (R, G, B), 0-255
_COLORMAPS = {
    "viridis": [
        (68, 1, 84),     # dark purple
        (59, 82, 139),   # blue
        (33, 145, 140),  # teal
        (94, 201, 97),   # green
        (253, 231, 37),  # yellow
    ],
    "magma": [
        (0, 0, 3),
        (49, 18, 59),
        (127, 39, 102),
        (196, 72, 60),
        (252, 253, 191),
    ],
    "coolwarm": [        # diverging: blue -> white -> red
        (59, 76, 192),
        (120, 141, 214),
        (190, 205, 232),
        (230, 230, 230),
        (222, 158, 114),
        (180, 4, 38),
    ],
}

_RESET = "\x1b[0m"

def _normalize_reducer_name(name: str) -> str:
    try:
        key = name.lower()
    except AttributeError as exc:
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

def _aggregate_block(block: np.ndarray, mode: str) -> float:
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

def _maybe_downsample_matrix(
    arr: np.ndarray, max_dim: int, reducer: str
) -> Tuple[np.ndarray, bool]:
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
            reduced[i, j] = _aggregate_block(block, reducer)
    return reduced, True

def _to_ndarray(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    # Handle PyTorch tensors (CPU/GPU, gradients)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach() if value.requires_grad else value
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            return tensor.numpy()
    except ImportError:
        pass

    # Handle TensorFlow tensors
    try:
        import tensorflow as tf  # type: ignore

        if isinstance(value, tf.Tensor):
            return value.numpy()
    except ImportError:
        pass

    # Generic array-like fallback
    try:
        return np.asarray(value)
    except Exception as exc:
        raise TypeError(
            "Unable to convert value to a NumPy array; ensure the input is array-like."
        ) from exc

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _interp_palette(palette, t: float) -> Tuple[int, int, int]:
    # t in [0,1]
    t = 0.0 if np.isnan(t) else min(1.0, max(0.0, float(t)))
    n = len(palette)
    if n == 1:
        return palette[0]
    pos = t * (n - 1)
    i = int(np.floor(pos))
    j = min(i + 1, n - 1)
    local_t = pos - i
    r = int(round(_lerp(palette[i][0], palette[j][0], local_t)))
    g = int(round(_lerp(palette[i][1], palette[j][1], local_t)))
    b = int(round(_lerp(palette[i][2], palette[j][2], local_t)))
    return (r, g, b)

def _ansi_bg(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"

def _ansi_fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"

def _luminance(r: int, g: int, b: int) -> float:
    # Relative luminance approximation for text contrast
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _format_number(x: float, precision: Optional[int], is_int: bool) -> str:
    if not np.isfinite(x):
        if np.isnan(x): return "NaN"
        return "+Inf" if x > 0 else "-Inf"
    if is_int:
        return str(int(round(x)))
    if precision is not None:
        return f"{x:.{precision}f}"
    ax = abs(x)
    if (ax != 0 and ax < 1e-3) or ax >= 1e4:
        return f"{x:.2e}"
    # default fixed with 3 decimals
    return f"{x:.3f}"

def print_matrix(
    arr: Any,
    *,
    cmap: str = "viridis",
    center_zero: bool = False,
    value_range: Optional[Tuple[float, float]] = None,  # (vmin, vmax)
    legend_bins: int = 5,
    precision: Optional[int] = None,
    na_bg: Tuple[int, int, int] = (128, 128, 128),
    na_fg: Tuple[int, int, int] = (255, 255, 255),
    show_values: bool = False,
    cell_char: str = " ",
    square_cells: bool = True,
    show_axes: bool = True,
    axis_tick_step: int = 5,
    max_display_size: int = 30,
    reducer: str = "average",
) -> None:
    """
    Pretty-print a 1D/2D array-like structure with colored cells, legend, and size.
    Cells render as single-character color blocks by default; enable show_values to overlay numbers.

    Parameters:
      arr: array-like (1D or 2D), real-valued. Works with Python lists, NumPy ndarrays,
           PyTorch/TensorFlow tensors (CPU), etc.
      cmap: 'viridis' | 'magma' | 'coolwarm' (can extend by editing _COLORMAPS).
      center_zero: if True, color scale is symmetric around 0.
      value_range: (vmin, vmax). If None, uses finite data min/max.
      legend_bins: number of color blocks in the legend.
      precision: decimals for floats. If None, choose automatically.
      na_bg: background color for NaN/Inf.
      na_fg: foreground color for NaN/Inf.
      show_values: if True, render numeric values inside each cell.
      cell_char: single-character string used when show_values is False.
      square_cells: if True, widen color cells (when values are hidden) to counteract terminal cell aspect ratio.
      show_axes: if True, draw central axes and annotate ticks.
      axis_tick_step: distance between tick labels along each axis (>=1).
      max_display_size: maximum visible size per dimension; arrays exceeding this are downsampled.
      reducer: 'average' | 'median' | 'max' | 'min' (plus aliases 'mean', 'avg', etc.) used during downsampling.
    """
    arr = _to_ndarray(arr)
    if not isinstance(cell_char, str) or len(cell_char) == 0:
        raise ValueError("cell_char must be a non-empty string.")
    cell_char_display = cell_char[0]
    if axis_tick_step <= 0:
        axis_tick_step = 0

    if arr.ndim == 1:
        arr2 = arr[np.newaxis, :]
    elif arr.ndim == 2:
        arr2 = arr
    else:
        raise ValueError("Only 1D or 2D arrays are supported for printing.")

    if cmap not in _COLORMAPS:
        raise ValueError(f"Unknown cmap '{cmap}'. Available: {list(_COLORMAPS)}")

    reducer_name = _normalize_reducer_name(reducer)
    orig_shape = arr2.shape
    orig_dtype = arr2.dtype
    arr2, _ = _maybe_downsample_matrix(arr2, max_display_size, reducer_name)
    display_shape = arr2.shape

    palette = _COLORMAPS[cmap]
    m, n = arr2.shape
    is_int_dtype = np.issubdtype(arr2.dtype, np.integer)

    finite_mask = np.isfinite(arr2)
    if value_range is None:
        if finite_mask.any():
            vmin = float(np.nanmin(arr2[finite_mask]))
            vmax = float(np.nanmax(arr2[finite_mask]))
            if center_zero:
                a = max(abs(vmin), abs(vmax))
                vmin, vmax = -a, a
            if vmin == vmax:
                # expand tiny range for visibility
                eps = 1.0 if vmin == 0 else abs(vmin) * 0.01 + 1e-9
                vmin -= eps
                vmax += eps
        else:
            # all non-finite
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = map(float, value_range)
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0

    formatted = None
    maxw = 0
    if show_values:
        formatted = np.empty_like(arr2, dtype=object)
        for i in range(m):
            for j in range(n):
                s = _format_number(arr2[i, j], precision, is_int_dtype)
                formatted[i, j] = s
                if len(s) > maxw:
                    maxw = len(s)
        cell_pad = 1  # one space left/right inside colored cell
        cell_width = 2 * cell_pad + maxw
    else:
        cell_pad = 0
        cell_width = 2 if square_cells else 1

    left_label_width = len(f"y={m - 1}") if show_axes and axis_tick_step and m > 0 else 0
    label_gap = 1 if show_axes and left_label_width > 0 else 0

    term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    total_width = (left_label_width + label_gap) + n * cell_width
    if n > 0 and total_width > term_width:
        print(f"[!] matrix width {total_width} exceeds terminal width {term_width}; output may wrap.")

    center_row = m // 2 if show_axes else -1
    center_col = n // 2 if show_axes else -1

    row_lines = []
    row_labels = []

    # Print rows with colored cells
    for i in range(m):
        line_parts = []
        for j in range(n):
            x = arr2[i, j]
            if np.isfinite(x):
                t = (float(x) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                r, g, b = _interp_palette(palette, t)
                bg = _ansi_bg(r, g, b)
                # High contrast text (black on bright bg, white otherwise)
                fg = _ansi_fg(0, 0, 0) if _luminance(r, g, b) > 140 else _ansi_fg(255, 255, 255)
            else:
                r, g, b = na_bg
                bg = _ansi_bg(*na_bg)
                fg = _ansi_fg(*na_fg)
            if show_values:
                s = formatted[i, j].rjust(maxw)
                content = f"{' ' * cell_pad}{s}{' ' * cell_pad}"
            else:
                content_char = cell_char_display
                if show_axes:
                    is_row_axis = i == center_row
                    is_col_axis = j == center_col
                    if is_row_axis and is_col_axis:
                        content_char = "+"
                    elif is_row_axis:
                        content_char = "-"
                    elif is_col_axis:
                        content_char = "|"
                content = content_char * max(1, cell_width)
            cell = f"{bg}{fg}{content}{_RESET}"
            line_parts.append(cell)
        row_line = "".join(line_parts)
        if show_axes and left_label_width:
            if axis_tick_step and (i % axis_tick_step == 0):
                label = f"y={i}".rjust(left_label_width)
            elif i == center_row:
                label = f"y={i}".rjust(left_label_width)
            else:
                label = " " * left_label_width
        else:
            label = " " * left_label_width if left_label_width else ""
        row_lines.append(row_line)
        row_labels.append(label)

    for idx, row_line in enumerate(row_lines):
        prefix = ""
        if left_label_width:
            prefix = f"{row_labels[idx]}{' ' if label_gap else ''}"
        print(prefix + row_line)

    if show_axes and n > 0 and axis_tick_step:
        prefix = " " * (left_label_width + label_gap)
        tick_cells = []
        for j in range(n):
            if j % axis_tick_step == 0:
                label = str(j)
                if len(label) > cell_width:
                    label = label[-cell_width:]
                tick_cells.append(label.center(cell_width))
            else:
                tick_cells.append(" " * cell_width)
        print(prefix + "".join(tick_cells))
        axis_caption = f"x-axis ticks (step={axis_tick_step})"
        if n * cell_width > 0:
            print(prefix + axis_caption[: n * cell_width])

    # Legend
    # Build color bar
    bins = max(2, int(legend_bins))
    bar_parts = []
    for k in range(bins):
        t = k / (bins - 1) if bins > 1 else 0.0
        r, g, b = _interp_palette(palette, t)
        bar_parts.append(f"{_ansi_bg(r, g, b)}  {_RESET}")  # two-space block
    print("Legend:", "".join(bar_parts))

    # Tick labels: show vmin, (0 if centered within range), vmax
    def _fmt_tick(v: float) -> str:
        if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3):
            return f"{v:.2e}"
        return f"{v:.4g}"

    tick_line = f"  min={_fmt_tick(vmin)}"
    if center_zero and vmin < 0 < vmax:
        tick_line += f", zero=0"
    tick_line += f", max={_fmt_tick(vmax)}"
    print(tick_line)

    # Size / dtype
    meta_parts = [f"shape={orig_shape}"]
    if display_shape != orig_shape:
        meta_parts.append(
            f"displayed_shape={display_shape} (reducer={reducer_name}, max_display_size={max_display_size})"
        )
    meta_parts.append(f"dtype={orig_dtype}")
    if arr2.dtype != orig_dtype:
        meta_parts.append(f"display_dtype={arr2.dtype}")
    meta_parts.append(f"finite_range=[{_fmt_tick(vmin)}, {_fmt_tick(vmax)}]")
    meta_parts.append(f"cmap={cmap}")
    meta_parts.append(f"center_zero={center_zero}")
    print(", ".join(meta_parts))
