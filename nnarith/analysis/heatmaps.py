from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn

from nnarith.analysis.base import Analysis
from nnarith.config import ArchConfig
from nnarith.history import HistoryRecord
from nnarith.records import ArtifactRecord, RunContext
from nnarith.utils import sanitize_label
from nnarith.visualization.terminal import print_matrix
from nnarith.encoding import decode_number, encode_number

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
class HeatmapAnalysisConfig:
    basename: str = "heatmaps"
    normalizer: HeatmapNormalizer = absolute_error_normalizer
    colormap: str = "viridis"
    store_raw_matrix: bool = True
    matrix_precision: Optional[int] = None
    preview_in_terminal: bool = False


class HeatmapAnalysis(Analysis):
    def __init__(self, config: HeatmapAnalysisConfig, results_dir: str) -> None:
        self._config = config
        self._results_dir = Path(results_dir)
        self._base_path = self._results_dir / config.basename
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._images_root = self._base_path / "images"
        self._images_root.mkdir(parents=True, exist_ok=True)
        self._records_root = self._base_path / "records"
        self._records_root.mkdir(parents=True, exist_ok=True)
        self._defaults: Dict[str, Optional[str]] = {
            "data": None,
            "training": None,
            "architecture": None,
        }
        self._order: Dict[str, List[str]] = {
            "data": [],
            "training": [],
            "architecture": [],
        }
        self._arch_defaults: Optional[Dict[str, object]] = None
        self._arch_orders: Dict[str, List[object]] = {
            "hidden_layers": [],
            "dropout": [],
            "l2": [],
        }
        self._operations: Tuple[Callable[[int, int], int], ...] = ()

    def begin_run(self, context: RunContext) -> None:
        self._register_value("data", context.data_name)
        self._register_value("training", context.training_name)
        self._operations = context.sweep.operations

    def observe_model(
        self,
        context: RunContext,
        arch_name: str,
        arch: ArchConfig,
        model: nn.Module,
        history: HistoryRecord,
    ) -> Sequence[ArtifactRecord]:
        artifacts: List[ArtifactRecord] = []
        data_index = self._register_value("data", context.data_name)
        training_index = self._register_value("training", context.training_name)
        arch_index = self._register_value("architecture", arch_name)
        arch_param_indices, arch_values = self._register_architecture_values(arch)

        operations = self._operations
        operand_digits = context.encoding.operand_digits
        base = context.encoding.base
        op_denominator = max(1, len(operations) - 1)
        left_values = list(
            range(context.combined_range.minimum, context.combined_range.maximum + 1)
        )
        right_values = list(left_values)

        model.eval()
        param = next(model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")

        operand_cache: Dict[int, List[float]] = {
            value: encode_number(value, operand_digits, base)
            for value in _distinct(left_values + right_values)
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
            op_segment = sanitize_label(op_name)
            previewed = False

            destinations = self._build_destinations(
                context=context,
                arch_name=arch_name,
                indices={
                    "data": data_index,
                    "training": training_index,
                    "architecture": arch_index,
                },
                arch_param_indices=arch_param_indices,
                arch_values=arch_values,
            )

            if self._config.preview_in_terminal and not previewed:
                print_matrix(
                    matrix,
                    cmap=self._config.colormap,
                    precision=self._config.matrix_precision,
                )
                previewed = True

            for destination in destinations:
                dimension = destination["dimension"]
                subdimension = destination.get("subdimension")
                value_label = destination["label"]
                stem_dimension = (
                    f"{dimension}" + (f"-{subdimension}" if subdimension else "")
                )
                stem = f"{destination['index']:03d}__{stem_dimension}={value_label}"
                image_dir = destination["images_dir"] / op_segment
                image_dir.mkdir(parents=True, exist_ok=True)
                plot_path = image_dir / f"{stem}.png"

                sweep_label = destination.get("subdimension") or destination["dimension"]
                fixed_parts = ", ".join(
                    f"{key}={val}" for key, val in destination["fixed"].items()
                )
                subtitle = (
                    f"{sweep_label} sweep value: {destination['value']} "
                    f"(fixed {fixed_parts})"
                )
                self._plot_heatmap(
                    matrix,
                    left_values,
                    right_values,
                    plot_path,
                    context,
                    arch_name,
                    op_name,
                    subtitle,
                )
                rel_plot_path = os.path.relpath(plot_path, context.results_dir)
                artifacts.append(
                    ArtifactRecord(
                        kind="heatmap_plot",
                        path=rel_plot_path,
                        metadata={
                            "dimension": destination["dimension"],
                            "parameter": destination.get("subdimension"),
                            "value": destination["value"],
                            "operation": op_name,
                            "arch": arch_name,
                            "data": context.data_name,
                            "training": context.training_name,
                        },
                    )
                )

                if self._config.store_raw_matrix:
                    record_dir = destination["records_dir"] / op_segment
                    record_dir.mkdir(parents=True, exist_ok=True)
                    record_path = record_dir / f"{stem}.json"
                    payload = {
                        "operation": op_name,
                        "left_operands": left_values,
                        "right_operands": right_values,
                        "normalized_values": matrix,
                        "normalizer": getattr(self._config.normalizer, "__name__", "custom"),
                        "train_range": context.train_range.to_dict(),
                        "combined_range": context.combined_range.to_dict(),
                        "sweep_dimension": destination["dimension"],
                        "sweep_value": destination["value"],
                        "fixed_parameters": destination["fixed"],
                    }
                    with record_path.open("w", encoding="utf-8") as handle:
                        json.dump(payload, handle, indent=2)
                    rel_data_path = os.path.relpath(record_path, context.results_dir)
                    artifacts.append(
                        ArtifactRecord(
                            kind="heatmap_matrix",
                            path=rel_data_path,
                            metadata={
                                "dimension": destination["dimension"],
                                "parameter": destination.get("subdimension"),
                                "value": destination["value"],
                                "operation": op_name,
                                "arch": arch_name,
                                "data": context.data_name,
                                "training": context.training_name,
                            },
                        )
                    )
        return artifacts

    def _register_value(self, category: str, value: str) -> int:
        order = self._order[category]
        if value not in order:
            order.append(value)
        if self._defaults[category] is None:
            self._defaults[category] = value
        return order.index(value) + 1

    def _register_architecture_values(
        self,
        arch: ArchConfig,
    ) -> Tuple[Dict[str, int], Dict[str, object]]:
        values: Dict[str, object] = {
            "hidden_layers": tuple(arch.hidden_layers),
            "dropout": arch.dropout,
            "l2": arch.l2,
        }
        indices: Dict[str, int] = {}
        for key, value in values.items():
            order = self._arch_orders[key]
            if value not in order:
                order.append(value)
            indices[key] = order.index(value) + 1
        if self._arch_defaults is None:
            self._arch_defaults = dict(values)
        return indices, values

    def _build_destinations(
        self,
        *,
        context: RunContext,
        arch_name: str,
        indices: Dict[str, int],
        arch_param_indices: Dict[str, int],
        arch_values: Dict[str, object],
    ) -> List[Dict[str, object]]:
        destinations: List[Dict[str, object]] = []
        defaults = self._defaults
        arch_defaults = self._arch_defaults or {}

        fixed_training = defaults["training"]
        fixed_arch = defaults["architecture"]
        fixed_data = defaults["data"]

        # Sweep over data (vary data, fix training + architecture)
        if (
            fixed_training is not None
            and fixed_arch is not None
            and context.training_name == fixed_training
            and arch_name == fixed_arch
        ):
            destinations.append(
                {
                    "dimension": "data",
                    "value": context.data_name,
                    "label": sanitize_label(context.data_name),
                    "index": indices["data"],
                    "fixed": {
                        "training": fixed_training,
                        "architecture": fixed_arch,
                    },
                    "images_dir": self._images_root / "data",
                    "records_dir": self._records_root / "data",
                }
            )

        # Sweep over training configs (vary training, fix data + architecture)
        if (
            fixed_data is not None
            and fixed_arch is not None
            and context.data_name == fixed_data
            and arch_name == fixed_arch
        ):
            destinations.append(
                {
                    "dimension": "training",
                    "value": context.training_name,
                    "label": sanitize_label(context.training_name),
                    "index": indices["training"],
                    "fixed": {
                        "data": fixed_data,
                        "architecture": fixed_arch,
                    },
                    "images_dir": self._images_root / "training",
                    "records_dir": self._records_root / "training",
                }
            )

        # Sweep over architectures (vary architecture, fix data + training)
        if (
            fixed_data is not None
            and fixed_training is not None
            and context.data_name == fixed_data
            and context.training_name == fixed_training
        ):
            destinations.append(
                {
                    "dimension": "architecture",
                    "subdimension": "name",
                    "value": arch_name,
                    "label": sanitize_label(arch_name),
                    "index": indices["architecture"],
                    "fixed": {
                        "data": fixed_data,
                        "training": fixed_training,
                    },
                    "images_dir": self._images_root / "architecture" / "name",
                    "records_dir": self._records_root / "architecture" / "name",
                }
            )

            for param, value in arch_values.items():
                default_value = arch_defaults.get(param)
                if default_value is None:
                    continue
                other_params_match = all(
                    arch_values[other] == arch_defaults.get(other)
                    for other in arch_values
                    if other != param
                )
                if not other_params_match:
                    continue
                destinations.append(
                    {
                        "dimension": "architecture",
                        "subdimension": param,
                        "value": value,
                        "label": sanitize_label(str(value)),
                        "index": arch_param_indices[param],
                        "fixed": {
                            "data": fixed_data,
                            "training": fixed_training,
                            **{
                                other: arch_defaults.get(other)
                                for other in arch_values
                                if other != param
                            },
                        },
                        "images_dir": self._images_root
                        / "architecture"
                        / sanitize_label(param),
                        "records_dir": self._records_root
                        / "architecture"
                        / sanitize_label(param),
                    }
                )

        return destinations

    def _plot_heatmap(
        self,
        matrix: Sequence[Sequence[float]],
        left_values: Sequence[int],
        right_values: Sequence[int],
        path: Path,
        context: RunContext,
        arch_name: str,
        op_name: str,
        subtitle: Optional[str] = None,
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
        title_lines = [
            f"{op_name} heatmap",
            f"data={context.data_name} | training={context.training_name} | arch={arch_name}",
        ]
        if subtitle:
            title_lines.append(subtitle)
        ax.set_title("\n".join(title_lines))
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


def _distinct(values: Iterable[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


__all__ = [
    "HeatmapAnalysis",
    "HeatmapAnalysisConfig",
    "HeatmapNormalizer",
    "absolute_error_normalizer",
]
