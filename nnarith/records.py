from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nnarith.config import ArchConfig, DataSweep, TrainingConfig
from nnarith.encoding import EncodingSpec
from nnarith.history import HistoryRecord, LossSeries, PlotRequest
from nnarith.utils import sanitize_label


@dataclass(frozen=True)
class RangeSummary:
    minimum: int
    maximum: int
    samples: Optional[int]

    @classmethod
    def from_split(cls, split: "SplitConfig") -> "RangeSummary":
        from nnarith.config import SplitConfig  # Local import to avoid circular import

        return cls(minimum=split.minimum, maximum=split.maximum, samples=split.samples)

    @classmethod
    def from_bounds(cls, minimum: int, maximum: int) -> "RangeSummary":
        return cls(minimum=minimum, maximum=maximum, samples=None)

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {"minimum": self.minimum, "maximum": self.maximum, "samples": self.samples}


@dataclass(frozen=True)
class TrainingConfigSummary:
    batch_size: int
    epochs: int
    learning_rate: float
    device: str

    @classmethod
    def from_config(cls, config: TrainingConfig, device: "torch.device") -> "TrainingConfigSummary":
        return cls(
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=str(device),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "device": self.device,
        }


@dataclass(frozen=True)
class HistorySummary:
    train: Tuple[float, ...]
    evaluations: Dict[str, Tuple[float, ...]]

    @classmethod
    def from_history(cls, history: HistoryRecord) -> "HistorySummary":
        return cls(
            train=tuple(history.train),
            evaluations={name: tuple(losses) for name, losses in history.evaluations.items()},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train": list(self.train),
            "evaluations": {name: list(losses) for name, losses in self.evaluations.items()},
        }


@dataclass(frozen=True)
class ArchitectureSummary:
    hidden_layers: Tuple[int, ...]
    dropout: float
    l2: float

    @classmethod
    def from_config(cls, config: ArchConfig) -> "ArchitectureSummary":
        return cls(
            hidden_layers=tuple(config.hidden_layers),
            dropout=config.dropout,
            l2=config.l2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_layers": list(self.hidden_layers),
            "dropout": self.dropout,
            "l2": self.l2,
        }


@dataclass
class ArtifactRecord:
    kind: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "path": self.path, "metadata": dict(self.metadata)}


@dataclass
class ArchitectureRecord:
    config: ArchitectureSummary
    history: HistorySummary
    final_train_loss: float
    final_eval_losses: Dict[str, float]
    mean_eval_loss: float
    artifacts: List[ArtifactRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "history": self.history.to_dict(),
            "final_train_loss": self.final_train_loss,
            "final_eval_losses": dict(self.final_eval_losses),
            "mean_eval_loss": self.mean_eval_loss,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass
class RunRecord:
    data_name: str
    training_name: str
    base: int
    operations: Tuple[str, ...]
    train_range: RangeSummary
    evaluation_ranges: Dict[str, RangeSummary]
    combined_range: RangeSummary
    training: TrainingConfigSummary
    architectures: Dict[str, ArchitectureRecord] = field(default_factory=dict)
    artifacts: List[ArtifactRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_name": self.data_name,
            "training_name": self.training_name,
            "base": self.base,
            "operations": list(self.operations),
            "train_range": self.train_range.to_dict(),
            "evaluation_ranges": {
                name: summary.to_dict() for name, summary in self.evaluation_ranges.items()
            },
            "combined_range": self.combined_range.to_dict(),
            "training": self.training.to_dict(),
            "architectures": {name: record.to_dict() for name, record in self.architectures.items()},
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


class RunRecorder:
    def __init__(self, results_dir: str, directory_name: str = "run_records") -> None:
        self._root = Path(results_dir)
        self._directory = self._root / directory_name
        self._directory.mkdir(parents=True, exist_ok=True)

    def write(self, record: RunRecord) -> Path:
        filename = f"{sanitize_label(record.data_name)}__{sanitize_label(record.training_name)}.json"
        path = self._directory / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2, sort_keys=True)
        return path


@dataclass(frozen=True)
class RunContext:
    data_name: str
    training_name: str
    sweep: DataSweep
    training_config: TrainingConfig
    encoding: EncodingSpec
    results_dir: str
    train_range: RangeSummary
    combined_range: RangeSummary


__all__ = [
    "ArchitectureRecord",
    "ArchitectureSummary",
    "ArtifactRecord",
    "HistorySummary",
    "RangeSummary",
    "RunContext",
    "RunRecord",
    "RunRecorder",
    "TrainingConfigSummary",
]
