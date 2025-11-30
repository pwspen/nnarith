from __future__ import annotations

from typing import Dict, Iterable, Tuple


_COLORMAPS: Dict[str, Tuple[Tuple[int, int, int], ...]] = {
    "viridis": (
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 97),
        (253, 231, 37),
    ),
    "magma": (
        (0, 0, 3),
        (49, 18, 59),
        (127, 39, 102),
        (196, 72, 60),
        (252, 253, 191),
    ),
    "coolwarm": (
        (59, 76, 192),
        (120, 141, 214),
        (190, 205, 232),
        (230, 230, 230),
        (222, 158, 114),
        (180, 4, 38),
    ),
}


def available_colormaps() -> Iterable[str]:
    return _COLORMAPS.keys()


def get_palette(name: str) -> Tuple[Tuple[int, int, int], ...]:
    try:
        return _COLORMAPS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown cmap '{name}'. Available: {list(_COLORMAPS)}") from exc


def lerp_channel(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def interpolate_palette(palette: Tuple[Tuple[int, int, int], ...], t: float) -> Tuple[int, int, int]:
    t = 0.0 if t != t else min(1.0, max(0.0, float(t)))  # NaN check: t != t
    n = len(palette)
    if n == 1:
        return palette[0]
    pos = t * (n - 1)
    i = int(pos)
    j = min(i + 1, n - 1)
    local_t = pos - i
    r = lerp_channel(palette[i][0], palette[j][0], local_t)
    g = lerp_channel(palette[i][1], palette[j][1], local_t)
    b = lerp_channel(palette[i][2], palette[j][2], local_t)
    return (r, g, b)


def ansi_bg(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"


def ansi_fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


def luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


__all__ = [
    "ansi_bg",
    "ansi_fg",
    "available_colormaps",
    "get_palette",
    "interpolate_palette",
    "lerp_channel",
    "luminance",
]
