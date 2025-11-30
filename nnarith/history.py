from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Sequence, Tuple

from nnarith.config import ArchConfig


@dataclass
class HistoryRecord:
    train: List[float]
    evaluations: Dict[str, List[float]]


@dataclass(frozen=True)
class LossSeries:
    label: str
    epochs: Tuple[int, ...]
    losses: Tuple[float, ...]
    metadata: Dict[str, str]


@dataclass(frozen=True)
class PlotRequest:
    data_name: str
    training_name: str
    series: Tuple[LossSeries, ...]


def history_to_series(
    arch_name: str,
    arch: ArchConfig,
    history: HistoryRecord,
    sweep_name: str,
    training_name: str,
) -> List[LossSeries]:
    """Convert a run history into plot-ready series with descriptive metadata."""
    base_metadata: Dict[str, str] = {
        "arch_name": arch_name,
        "sweep": sweep_name,
        "training": training_name,
    }
    for field in fields(arch):
        base_metadata[field.name] = repr(getattr(arch, field.name))

    series_records: List[LossSeries] = []
    if history.train:
        epochs = tuple(range(1, len(history.train) + 1))
        series_records.append(
            LossSeries(
                label=f"{arch_name} train",
                epochs=epochs,
                losses=tuple(history.train),
                metadata={**base_metadata, "series_kind": "train", "split": "train"},
            )
        )
    for eval_name, losses in history.evaluations.items():
        if not losses:
            continue
        epochs = tuple(range(1, len(losses) + 1))
        series_records.append(
            LossSeries(
                label=f"{arch_name} {eval_name}",
                epochs=epochs,
                losses=tuple(losses),
                metadata={**base_metadata, "series_kind": "evaluation", "split": eval_name},
            )
        )
    return series_records


__all__ = ["HistoryRecord", "LossSeries", "PlotRequest", "history_to_series"]
