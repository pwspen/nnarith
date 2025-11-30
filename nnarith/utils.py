from __future__ import annotations

from typing import Optional


def sanitize_label(label: str) -> str:
    """Replace unsupported characters so the value becomes filename-friendly."""
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in label)


def plot_filename(
    base: str,
    data_name: str,
    training_name: str,
    group_label: Optional[str] = None,
) -> str:
    """Build a sanitized filename stem for plot artifacts."""
    suffix = f"__{sanitize_label(group_label)}" if group_label else ""
    return f"{base}_{sanitize_label(data_name)}__{sanitize_label(training_name)}{suffix}.png"


__all__ = ["plot_filename", "sanitize_label"]
