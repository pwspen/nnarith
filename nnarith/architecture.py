from __future__ import annotations

from typing import Dict, Sequence

from nnarith.config import ArchConfig


def enumerate_architectures(
    layer_options: Sequence[Sequence[int]],
    dropouts: Sequence[float],
    l2_penalties: Sequence[float],
) -> Dict[str, ArchConfig]:
    """Expand layer/dropout/L2 grids into a named architecture dictionary."""
    architectures: Dict[str, ArchConfig] = {}
    for layers in layer_options:
        layer_tuple = tuple(layers)
        for dropout in dropouts:
            for l2 in l2_penalties:
                label = f"layers={list(layer_tuple)} dropout={dropout} l2={l2}"
                architectures[label] = ArchConfig(hidden_layers=layer_tuple, dropout=dropout, l2=l2)
    return architectures


__all__ = ["enumerate_architectures"]
