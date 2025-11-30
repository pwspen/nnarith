from __future__ import annotations

from typing import Callable, List, Sequence

from torch import nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: Sequence[int],
    dropout: float = 0.0,
    activation: Callable[[], nn.Module] = nn.ReLU,
) -> nn.Module:
    layers: List[nn.Module] = []
    previous = input_dim
    for width in hidden_layers:
        layers.append(nn.Linear(previous, width))
        layers.append(activation())
        previous = width
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


__all__ = ["build_mlp"]
