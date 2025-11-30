from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from gen import EncodingSpec, decode_number, encode_number, generate_dataset_for_range
from print_matrix import print_matrix

def add(left: int, right: int) -> int:
    return left + right

def subtract(left: int, right: int) -> int:
    return left - right

def multiply(left: int, right: int) -> int:
    return left * right


def sanitize_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in label)


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


HeatmapNormalizer = Callable[[int, int, Callable[[int, int], int], int, int], float]


def absolute_error_normalizer(
    left: int,
    right: int,
    operation: Callable[[int, int], int],
    prediction: int,
    target: int,
) -> float:
    _ = (left, right, operation)
    return float(abs(prediction - target))


@dataclass(frozen=True)
class RangeSummary:
    minimum: int
    maximum: int
    samples: Optional[int]

    @classmethod
    def from_split(cls, split: SplitConfig) -> "RangeSummary":
        return cls(minimum=split.minimum, maximum=split.maximum, samples=split.samples)

    @classmethod
    def from_bounds(cls, minimum: int, maximum: int) -> "RangeSummary":
        return cls(minimum=minimum, maximum=maximum, samples=None)

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class TrainingConfigSummary:
    batch_size: int
    epochs: int
    learning_rate: float
    device: str

    @classmethod
    def from_config(cls, config: TrainingConfig, device: torch.device) -> "TrainingConfigSummary":
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
class ArchConfig:
    hidden_layers: Sequence[int]
    dropout: float = 0.0
    l2: float = 0.0

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
        return {
            "kind": self.kind,
            "path": self.path,
            "metadata": dict(self.metadata),
        }


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
            "architectures": {
                name: record.to_dict() for name, record in self.architectures.items()
            },
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


class Analysis:
    def begin_run(self, context: RunContext) -> None:
        return None

    def observe_model(
        self,
        context: RunContext,
        arch_name: str,
        arch: ArchConfig,
        model: nn.Module,
        history: HistoryRecord,
    ) -> Sequence[ArtifactRecord]:
        return ()

    def finalize_run(self, context: RunContext) -> Sequence[ArtifactRecord]:
        return ()

    def finalize_experiment(self) -> None:
        return None


@dataclass(frozen=True)
class LossCurveAnalysisConfig:
    basename: str = "loss_curves"
    group_by: Tuple[str, ...] = ("hidden_layers",)
    base_selectors: Optional[Sequence[Dict[str, str]]] = None


class LossCurveAnalysis(Analysis):
    def __init__(self, config: LossCurveAnalysisConfig, results_dir: str) -> None:
        self._config = config
        self._results_dir = results_dir
        self._plot_base = os.path.join(results_dir, config.basename)
        self._plot_requests: List[PlotRequest] = []
        self._global_epoch_max = 0
        self._global_loss_min = float("inf")
        self._global_loss_max = float("-inf")
        self._current_series: List[LossSeries] = []
        self._current_run_key: Optional[Tuple[str, str]] = None
        self._artifacts_by_run: Dict[Tuple[str, str], List[ArtifactRecord]] = {}

    def begin_run(self, context: RunContext) -> None:
        self._current_series = []
        self._current_run_key = (context.data_name, context.training_name)
        self._artifacts_by_run.setdefault(self._current_run_key, [])

    def observe_model(
        self,
        context: RunContext,
        arch_name: str,
        arch: ArchConfig,
        model: nn.Module,
        history: HistoryRecord,
    ) -> Sequence[ArtifactRecord]:
        series_records = history_to_series(
            arch_name=arch_name,
            arch=arch,
            history=history,
            sweep_name=context.data_name,
            training_name=context.training_name,
        )
        self._current_series.extend(series_records)
        for record in series_records:
            if record.losses:
                self._global_epoch_max = max(self._global_epoch_max, len(record.losses))
                self._global_loss_min = min(self._global_loss_min, min(record.losses))
                self._global_loss_max = max(self._global_loss_max, max(record.losses))
        return ()

    def finalize_run(self, context: RunContext) -> Sequence[ArtifactRecord]:
        usable_series = [series for series in self._current_series if series.losses]
        if not usable_series:
            return ()
        self._plot_requests.append(
            PlotRequest(
                data_name=context.data_name,
                training_name=context.training_name,
                series=tuple(usable_series),
            )
        )
        return ()

    @property
    def artifacts_by_run(self) -> Dict[Tuple[str, str], List[ArtifactRecord]]:
        return self._artifacts_by_run

    def finalize_experiment(self) -> None:
        if not self._plot_requests:
            return

        global_epoch_max = max(self._global_epoch_max, 1)
        x_limits = (1, global_epoch_max)

        loss_min = self._global_loss_min if self._global_loss_min != float("inf") else 0.0
        loss_max = self._global_loss_max if self._global_loss_max != float("-inf") else 1.0

        loss_span = max(loss_max - loss_min, 1e-8)
        margin = loss_span * 0.05
        y_limits = (loss_min - margin, loss_max + margin)

        for request in self._plot_requests:
            grouped_series = group_series_by_keys(request.series, self._config.group_by)
            base_series: List[LossSeries] = []
            if self._config.base_selectors:
                for selector in self._config.base_selectors:
                    for record in request.series:
                        if all(record.metadata.get(key) == value for key, value in selector.items()):
                            base_series.append(record)

            unique_base_series: List[LossSeries] = []
            seen_labels: Set[str] = set()
            for record in base_series:
                if record.label in seen_labels:
                    continue
                unique_base_series.append(record)
                seen_labels.add(record.label)

            for group_key, series in grouped_series.items():
                group_label = (
                    ", ".join(
                        f"{key}={value}" for key, value in zip(self._config.group_by, group_key)
                    )
                    if self._config.group_by
                    else None
                )
                combined_series: List[LossSeries] = []
                seen: Set[str] = set()

                for record in unique_base_series:
                    if record.label in seen:
                        continue
                    combined_series.append(record)
                    seen.add(record.label)

                contributed_non_base = False
                base_labels = {record.label for record in unique_base_series}
                for record in series:
                    if record.label in seen:
                        continue
                    combined_series.append(record)
                    seen.add(record.label)
                    if record.label not in base_labels:
                        contributed_non_base = True

                if not contributed_non_base and unique_base_series:
                    continue

                path = plot_filename(
                    self._plot_base, request.data_name, request.training_name, group_label
                )
                plot_series_group(
                    combined_series,
                    path,
                    request.data_name,
                    request.training_name,
                    group_label,
                    x_limits,
                    y_limits,
                )
                print(f"Saved loss curves to {path}")
                run_key = (request.data_name, request.training_name)
                rel_path = os.path.relpath(path, self._results_dir)
                artifact = ArtifactRecord(
                    kind="loss_curve_plot",
                    path=rel_path,
                    metadata={"group": group_label} if group_label else {},
                )
                self._artifacts_by_run.setdefault(run_key, []).append(artifact)


@dataclass(frozen=True)
class HeatmapAnalysisConfig:
    basename: str = "heatmaps"
    normalizer: HeatmapNormalizer = absolute_error_normalizer
    colormap: str = "viridis"
    store_raw_matrix: bool = True


class HeatmapAnalysis(Analysis):
    def __init__(self, config: HeatmapAnalysisConfig, results_dir: str) -> None:
        self._config = config
        self._results_dir = Path(results_dir)
        self._base_path = self._results_dir / config.basename
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._current_run_dir: Optional[Path] = None
        self._operations: Tuple[Callable[[int, int], int], ...] = ()

    def begin_run(self, context: RunContext) -> None:
        run_dir_name = (
            f"{sanitize_label(context.data_name)}__{sanitize_label(context.training_name)}"
        )
        self._current_run_dir = self._base_path / run_dir_name
        self._current_run_dir.mkdir(parents=True, exist_ok=True)
        self._operations = context.sweep.operations

    def observe_model(
        self,
        context: RunContext,
        arch_name: str,
        arch: ArchConfig,
        model: nn.Module,
        history: HistoryRecord,
    ) -> Sequence[ArtifactRecord]:
        if self._current_run_dir is None:
            return ()
        artifacts: List[ArtifactRecord] = []
        operations = self._operations
        operand_digits = context.encoding.operand_digits
        result_digits = context.encoding.result_digits
        base = context.encoding.base
        op_denominator = max(1, len(operations) - 1)
        left_values = list(range(context.combined_range.minimum, context.combined_range.maximum + 1))
        right_values = list(left_values)

        model.eval()
        device = next(model.parameters()).device

        operand_cache: Dict[int, List[float]] = {
            value: encode_number(value, operand_digits, base) for value in set(left_values + right_values)
        }

        for op_index, operation in enumerate(operations):
            features: List[List[float]] = []
            targets: List[int] = []
            coordinate_pairs: List[Tuple[int, int]] = []
            for left in left_values:
                for right in right_values:
                    feature = (
                        operand_cache[left]
                        + operand_cache[right]
                        + [op_index / op_denominator]
                    )
                    features.append(feature)
                    targets.append(operation(left, right))
                    coordinate_pairs.append((left, right))

            if not features:
                continue

            inputs = torch.tensor(features, dtype=torch.float32, device=device)
            with torch.no_grad():
                outputs = model(inputs)

            predicted_numbers: List[int] = []
            for index in range(outputs.size(0)):
                prediction = outputs[index].cpu().tolist()
                decoded = decode_number(prediction, base)
                predicted_numbers.append(decoded)

            normalized_values: List[float] = []
            for (left, right), prediction, target in zip(
                coordinate_pairs, predicted_numbers, targets
            ):
                normalized_values.append(
                    self._config.normalizer(left, right, operation, prediction, target)
                )

            matrix: List[List[float]] = []
            idx = 0
            for _ in left_values:
                row = normalized_values[idx : idx + len(right_values)]
                matrix.append(row)
                idx += len(right_values)

            op_name = getattr(operation, "__name__", f"op_{op_index}")
            stem = (
                f"{sanitize_label(arch_name)}__{sanitize_label(op_name)}"
            )
            plot_path = self._current_run_dir / f"{stem}.png"
            print_matrix(matrix)
            self._plot_heatmap(
                matrix,
                left_values,
                right_values,
                plot_path,
                context,
                arch_name,
                op_name,
            )
            rel_plot_path = os.path.relpath(plot_path, context.results_dir)
            artifacts.append(
                ArtifactRecord(
                    kind="heatmap_plot",
                    path=rel_plot_path,
                    metadata={
                        "arch": arch_name,
                        "operation": op_name,
                    },
                )
            )

            if self._config.store_raw_matrix:
                data_path = self._current_run_dir / f"{stem}.json"
                payload = {
                    "operation": op_name,
                    "left_operands": left_values,
                    "right_operands": right_values,
                    "normalized_values": matrix,
                    "normalizer": getattr(self._config.normalizer, "__name__", "custom"),
                    "train_range": context.train_range.to_dict(),
                    "combined_range": context.combined_range.to_dict(),
                }
                with data_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                rel_data_path = os.path.relpath(data_path, context.results_dir)
                artifacts.append(
                    ArtifactRecord(
                        kind="heatmap_matrix",
                        path=rel_data_path,
                        metadata={
                            "arch": arch_name,
                            "operation": op_name,
                        },
                    )
                )
        return artifacts

    def _plot_heatmap(
        self,
        matrix: Sequence[Sequence[float]],
        left_values: Sequence[int],
        right_values: Sequence[int],
        path: Path,
        context: RunContext,
        arch_name: str,
        op_name: str,
    ) -> None:
        import numpy as np
        import matplotlib.patches as patches

        data = np.array(matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        extent = (
            context.combined_range.minimum - 0.5,
            context.combined_range.maximum + 0.5,
            context.combined_range.minimum - 0.5,
            context.combined_range.maximum + 0.5,
        )
        im = ax.imshow(
            data,
            origin="lower",
            interpolation="nearest",
            cmap=self._config.colormap,
            extent=extent,
            aspect="auto",
        )
        ax.set_xlabel("Right operand")
        ax.set_ylabel("Left operand")
        ax.set_title(
            f"{op_name} heatmap\narch={arch_name}\ndata={context.data_name} | training={context.training_name}"
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized error")

        train_min = context.train_range.minimum - 0.5
        train_span = context.train_range.maximum - context.train_range.minimum + 1
        rect = patches.Rectangle(
            (train_min, train_min),
            train_span,
            train_span,
            linewidth=1.2,
            edgecolor="white",
            facecolor="none",
            linestyle="--",
            label="Train region",
        )
        ax.add_patch(rect)
        ax.legend(loc="upper right", frameon=False)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


@dataclass
class AnalysisPlan:
    loss_curves: Optional[LossCurveAnalysisConfig] = None
    network_heatmaps: Optional[HeatmapAnalysisConfig] = None

def resolve_device(training_config: TrainingConfig) -> torch.device:
    device = training_config.device
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_filename(base: str, data_name: str, training_name: str, group_label: Optional[str] = None) -> str:
    suffix = f"__{sanitize_label(group_label)}" if group_label else ""
    return f"{base}_{sanitize_label(data_name)}__{sanitize_label(training_name)}{suffix}.png"


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


def required_digits(max_abs_value: int, base: int) -> int:
    digits = 1
    threshold = base
    while max_abs_value >= threshold:
        digits += 1
        threshold *= base
    return digits


def compute_encoding(sweep: DataSweep) -> EncodingSpec:
    candidates = {sweep.train.minimum, sweep.train.maximum}
    for split in sweep.evaluations.values():
        candidates.add(split.minimum)
        candidates.add(split.maximum)

    operand_max_abs = max(abs(value) for value in candidates)

    result_max_abs = 0
    for left in candidates:
        for right in candidates:
            for operation in sweep.operations:
                result_max_abs = max(result_max_abs, abs(operation(left, right)))

    operand_digits = required_digits(operand_max_abs, sweep.base)
    result_digits = required_digits(result_max_abs, sweep.base)
    return EncodingSpec(
        base=sweep.base,
        operand_digits=operand_digits,
        result_digits=result_digits,
    )

def history_to_series(
    arch_name: str,
    arch: ArchConfig,
    history: HistoryRecord,
    sweep_name: str,
    training_name: str,
) -> List[LossSeries]:
    """
    Convert a run history into plot-ready series with descriptive metadata.
    """
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


def enumerate_architectures(
    layer_options: Sequence[Sequence[int]],
    dropouts: Sequence[float],
    l2_penalties: Sequence[float],
) -> Dict[str, ArchConfig]:
    """
    Expand layer/dropout/L2 grids into a named architecture dictionary.
    """
    architectures: Dict[str, ArchConfig] = {}
    for layers in layer_options:
        layer_tuple = tuple(layers)
        for dropout in dropouts:
            for l2 in l2_penalties:
                label = f"layers={list(layer_tuple)} dropout={dropout} l2={l2}"
                architectures[label] = ArchConfig(hidden_layers=layer_tuple, dropout=dropout, l2=l2)
    return architectures


def group_series_by_keys(
    series: Sequence[LossSeries],
    keys: Sequence[str],
) -> Dict[Tuple[str, ...], List[LossSeries]]:
    if not keys:
        return {(): list(series)}

    grouped: Dict[Tuple[str, ...], List[LossSeries]] = {}
    for record in series:
        key_values: List[str] = []
        missing: List[str] = []
        for key in keys:
            value = record.metadata.get(key)
            if value is None:
                missing.append(key)
                continue
            key_values.append(str(value))
        if missing:
            missing_keys = ", ".join(missing)
            raise KeyError(f"Missing metadata {missing_keys} for series {record.label}")
        grouped.setdefault(tuple(key_values), []).append(record)
    return grouped


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_examples = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size
    return running_loss / max(1, total_examples)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = criterion(predictions, targets)
            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size
    return running_loss / max(1, total_examples)

def run_experiments(
    data_sweeps: Dict[str, DataSweep],
    training_configs: Dict[str, TrainingConfig],
    architectures: Dict[str, ArchConfig],
    *,
    results_dir: str = "results",
    torch_seed: Optional[int] = 0,
    analysis_plan: Optional[AnalysisPlan] = None,
) -> None:
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    os.makedirs(results_dir, exist_ok=True)
    if analysis_plan is None:
        analysis_plan = AnalysisPlan(loss_curves=LossCurveAnalysisConfig())

    analyses: List[Analysis] = []
    loss_curve_analysis: Optional[LossCurveAnalysis] = None
    if analysis_plan.loss_curves is not None:
        loss_curve_analysis = LossCurveAnalysis(analysis_plan.loss_curves, results_dir)
        analyses.append(loss_curve_analysis)
    if analysis_plan.network_heatmaps is not None:
        analyses.append(HeatmapAnalysis(analysis_plan.network_heatmaps, results_dir))

    recorder = RunRecorder(results_dir)
    run_records: Dict[Tuple[str, str], RunRecord] = {}

    for sweep_name, sweep in data_sweeps.items():
        encoding = compute_encoding(sweep)
        train_dataset, _, train_operand_max, train_result_max = generate_dataset_for_range(
            sweep.train.minimum,
            sweep.train.maximum,
            sweep.operations,
            sweep.base,
            samples=sweep.train.samples,
            seed=sweep.seed,
            encoding=encoding,
        )

        print(f"\n=== Data sweep: {sweep_name} ===")
        print(
            f"Train range=[{sweep.train.minimum}, {sweep.train.maximum}] "
            f"samples={sweep.train.samples}"
        )
        for eval_name, split in sweep.evaluations.items():
            print(
                f"  Eval[{eval_name}] range=[{split.minimum}, {split.maximum}] "
                f"samples={split.samples}"
            )
        print(
            f"Operand digits={encoding.operand_digits}, "
            f"Result digits={encoding.result_digits}, "
            f"input_size={encoding.input_size}, target_size={encoding.target_size}"
        )

        eval_datasets: Dict[str, Tuple[Dataset, int, int]] = {}
        for index, (eval_name, split) in enumerate(sweep.evaluations.items(), start=1):
            eval_seed = sweep.seed + index
            dataset, _, operand_max_abs, result_max_abs = generate_dataset_for_range(
                split.minimum,
                split.maximum,
                sweep.operations,
                sweep.base,
                samples=split.samples,
                seed=eval_seed,
                encoding=encoding,
            )
            eval_datasets[eval_name] = (dataset, operand_max_abs, result_max_abs)

        dataset_operand_max = max(
            [train_operand_max, *[stats[1] for stats in eval_datasets.values()]]
        )
        dataset_result_max = max(
            [train_result_max, *[stats[2] for stats in eval_datasets.values()]]
        )
        print(f"operand_max_abs={dataset_operand_max}, result_max_abs={dataset_result_max}")

        train_range_summary = RangeSummary.from_split(sweep.train)
        evaluation_summaries = {
            name: RangeSummary.from_split(split) for name, split in sweep.evaluations.items()
        }
        combined_min = sweep.train.minimum
        combined_max = sweep.train.maximum
        for split in sweep.evaluations.values():
            combined_min = min(combined_min, split.minimum)
            combined_max = max(combined_max, split.maximum)
        combined_range_summary = RangeSummary.from_bounds(combined_min, combined_max)
        operation_names = tuple(
            getattr(operation, "__name__", repr(operation)) for operation in sweep.operations
        )

        for training_name, training_config in training_configs.items():
            device = resolve_device(training_config)

            print(f"\n--- Training config: {training_name} (device={device}) ---")

            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
            )
            evaluation_loaders: Dict[str, DataLoader] = {
                eval_name: DataLoader(
                    dataset,
                    batch_size=training_config.batch_size,
                    shuffle=False,
                )
                for eval_name, (dataset, _, _) in eval_datasets.items()
            }

            criterion = nn.MSELoss()
            results = []

            training_summary = TrainingConfigSummary.from_config(training_config, device)
            run_context = RunContext(
                data_name=sweep_name,
                training_name=training_name,
                sweep=sweep,
                training_config=training_config,
                encoding=encoding,
                results_dir=results_dir,
                train_range=train_range_summary,
                combined_range=combined_range_summary,
            )
            for analysis in analyses:
                analysis.begin_run(run_context)

            run_key = (sweep_name, training_name)
            run_record = RunRecord(
                data_name=sweep_name,
                training_name=training_name,
                base=sweep.base,
                operations=operation_names,
                train_range=train_range_summary,
                evaluation_ranges=dict(evaluation_summaries),
                combined_range=combined_range_summary,
                training=training_summary,
            )
            run_records[run_key] = run_record

            for arch_name, arch in architectures.items():
                model = build_mlp(
                    encoding.input_size,
                    encoding.target_size,
                    arch.hidden_layers,
                    dropout=arch.dropout,
                ).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=training_config.learning_rate, weight_decay=arch.l2
                )
                history = HistoryRecord(
                    train=[],
                    evaluations={name: [] for name in evaluation_loaders},
                )
                for epoch in range(training_config.epochs):
                    train_loss = train_epoch(
                        model, train_loader, optimizer, criterion, device
                    )
                    history.train.append(train_loss)
                    eval_losses: Dict[str, float] = {}
                    for eval_name, loader in evaluation_loaders.items():
                        loss_value = evaluate(model, loader, criterion, device)
                        history.evaluations[eval_name].append(loss_value)
                        eval_losses[eval_name] = loss_value
                    eval_summary = ", ".join(
                        f"{name}={loss:.6f}" for name, loss in eval_losses.items()
                    )
                    suffix = f" eval[{eval_summary}]" if eval_summary else ""
                    print(
                        f"{arch_name} ({sweep_name} | {training_name}): "
                        f"epoch={epoch + 1} train_loss={train_loss:.6f}{suffix}"
                    )
                final_train_loss = history.train[-1] if history.train else float("inf")
                final_eval_losses = {
                    name: losses[-1] for name, losses in history.evaluations.items()
                }
                mean_eval_loss = (
                    sum(final_eval_losses.values()) / max(1, len(final_eval_losses))
                    if final_eval_losses
                    else float("inf")
                )
                results.append((arch_name, final_train_loss, final_eval_losses, mean_eval_loss))

                history_summary = HistorySummary.from_history(history)
                arch_summary = ArchitectureSummary.from_config(arch)
                arch_record = ArchitectureRecord(
                    config=arch_summary,
                    history=history_summary,
                    final_train_loss=final_train_loss,
                    final_eval_losses=final_eval_losses,
                    mean_eval_loss=mean_eval_loss,
                )

                for analysis in analyses:
                    arch_artifacts = analysis.observe_model(
                        run_context, arch_name, arch, model, history
                    )
                    if arch_artifacts:
                        arch_record.artifacts.extend(arch_artifacts)

                run_record.architectures[arch_name] = arch_record

            if results:
                ranked = sorted(results, key=lambda item: item[3])
                print(f"\nRanked by mean eval loss ({sweep_name} | {training_name}):")
                for rank, (name, train_loss, eval_losses, _) in enumerate(
                    ranked, start=1
                ):
                    eval_report = ", ".join(
                        f"{eval_name}={loss:.6f}" for eval_name, loss in eval_losses.items()
                    )
                    print(
                        f"{rank}. {name}: train_loss={train_loss:.6f}"
                        + (f", eval[{eval_report}]" if eval_report else "")
                    )

            run_level_artifacts: List[ArtifactRecord] = []
            for analysis in analyses:
                run_level_artifacts.extend(analysis.finalize_run(run_context))
            if run_level_artifacts:
                run_record.artifacts.extend(run_level_artifacts)

    for analysis in analyses:
        analysis.finalize_experiment()

    if loss_curve_analysis is not None:
        for run_key, artifacts in loss_curve_analysis.artifacts_by_run.items():
            if not artifacts:
                continue
            record = run_records.get(run_key)
            if record is not None:
                record.artifacts.extend(artifacts)

    for run_key in sorted(run_records):
        recorder.write(run_records[run_key])


def plot_series_group(
    series: Sequence[LossSeries],
    path: str,
    data_name: str,
    training_name: str,
    group_label: Optional[str],
    x_limits: Tuple[int, int],
    y_limits: Tuple[float, float],
) -> None:
    plt.figure(figsize=(8, 5))
    color_cycle = plt.cm.tab10.colors
    dash_styles = ["--", "-.", ":"]

    eval_order: List[str] = []
    for record in series:
        if record.metadata.get("series_kind") == "evaluation":
            split_name = record.metadata.get("split", "")
            if split_name not in eval_order:
                eval_order.append(split_name)

    color_map: Dict[str, Tuple[float, float, float]] = {}

    for record in series:
        arch_name = record.metadata.get("arch_name", record.label)
        if arch_name not in color_map:
            color_map[arch_name] = color_cycle[len(color_map) % len(color_cycle)]

        color = color_map[arch_name]
        series_kind = record.metadata.get("series_kind", "")
        linestyle = "-"
        linewidth = 1.8 if series_kind == "train" else 1.5
        if series_kind != "train":
            split_name = record.metadata.get("split", "")
            if split_name not in eval_order:
                eval_order.append(split_name)
            eval_index = eval_order.index(split_name)
            linestyle = dash_styles[eval_index % len(dash_styles)]

        plt.plot(
            record.epochs,
            record.losses,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=record.label,
        )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    title = f"Training vs Eval Loss\n(data={data_name}, training={training_name})"
    if group_label:
        title += f"\nGroup: {group_label}"
    plt.title(title)
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.xlim(*x_limits)
    plt.ylim(*y_limits)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
