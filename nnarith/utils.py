from __future__ import annotations

import os
import re
from typing import Optional


def sanitize_label(label: Optional[str]) -> str:
    """Replace unsupported characters so the value becomes filename-friendly."""
    if label is None:
        return "item"
    tokens = re.findall(r"[0-9A-Za-z]+", label)
    if not tokens:
        return "item"
    return "-".join(token.lower() for token in tokens)


def plot_filename(
    base: str,
    data_name: str,
    training_name: str,
    group_label: Optional[str] = None,
) -> str:
    """Build a sanitized filename stem for plot artifacts."""
    data_segment = sanitize_label(data_name)
    training_segment = sanitize_label(training_name)
    suffix = f"__{sanitize_label(group_label)}" if group_label else ""
    directory = os.path.join(base, data_segment)
    filename = f"{training_segment}{suffix}.png"
    return os.path.join(directory, filename)


__all__ = ["plot_filename", "sanitize_label"]
