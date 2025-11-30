from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SplitConfig:
    minimum: int
    maximum: int
    samples: Optional[int] = None


@dataclass(frozen=True)
class DataSweep:
    base: int
    train: SplitConfig
    evaluations: Dict[str, SplitConfig]
    seed: int
    operations: Tuple[Callable[[int, int], int], ...]


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    device: Optional[torch.device] = None


@dataclass(frozen=True)
class ArchConfig:
    hidden_layers: Sequence[int]
    dropout: float = 0.0
    l2: float = 0.0


__all__ = ["ArchConfig", "DataSweep", "SplitConfig", "TrainingConfig"]
