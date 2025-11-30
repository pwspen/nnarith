from __future__ import annotations

import shutil
from typing import Any, Optional, Tuple

import numpy as np

from nnarith.visualization.helpers import (
    format_number,
    maybe_downsample_matrix,
    normalize_reducer_name,
    to_ndarray,
)
from nnarith.visualization.palette import (
    ansi_bg,
    ansi_fg,
    available_colormaps,
    get_palette,
    interpolate_palette,
    luminance,
)

_RESET = "\x1b[0m"


def print_matrix(
    arr: Any,
    *,
    cmap: str = "viridis",
    center_zero: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
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
    """
    arr = to_ndarray(arr)
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

    colormaps = tuple(available_colormaps())
    if cmap not in colormaps:
        raise ValueError(f"Unknown cmap '{cmap}'. Available: {list(colormaps)}")

    reducer_name = normalize_reducer_name(reducer)
    orig_shape = arr2.shape
    orig_dtype = arr2.dtype
    arr2, _ = maybe_downsample_matrix(arr2, max_display_size, reducer_name)
    display_shape = arr2.shape

    palette = get_palette(cmap)
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
                eps = 1.0 if vmin == 0 else abs(vmin) * 0.01 + 1e-9
                vmin -= eps
                vmax += eps
        else:
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
                s = format_number(arr2[i, j], precision, is_int_dtype)
                formatted[i, j] = s
                if len(s) > maxw:
                    maxw = len(s)
        cell_pad = 1
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

    for i in range(m):
        line_parts = []
        for j in range(n):
            x = arr2[i, j]
            if np.isfinite(x):
                t = (float(x) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                r, g, b = interpolate_palette(palette, t)
                bg = ansi_bg(r, g, b)
                fg = ansi_fg(0, 0, 0) if luminance(r, g, b) > 140 else ansi_fg(255, 255, 255)
            else:
                r, g, b = na_bg
                bg = ansi_bg(*na_bg)
                fg = ansi_fg(*na_fg)
            if show_values and formatted is not None:
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
            if axis_tick_step and (i % axis_tick_step == 0 or i == center_row):
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

    bins = max(2, int(legend_bins))
    bar_parts = []
    for k in range(bins):
        t = k / (bins - 1) if bins > 1 else 0.0
        r, g, b = interpolate_palette(palette, t)
        bar_parts.append(f"{ansi_bg(r, g, b)}  {_RESET}")
    print("Legend:", "".join(bar_parts))

    def _fmt_tick(v: float) -> str:
        if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3):
            return f"{v:.2e}"
        return f"{v:.4g}"

    tick_line = f"  min={_fmt_tick(vmin)}"
    if center_zero and vmin < 0 < vmax:
        tick_line += ", zero=0"
    tick_line += f", max={_fmt_tick(vmax)}"
    print(tick_line)

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


__all__ = ["print_matrix"]
