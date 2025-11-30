from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn

from gen import decode_number, encode_number

from nnarith.analysis.base import Analysis
from nnarith.config import ArchConfig
from nnarith.history import HistoryRecord
from nnarith.records import ArtifactRecord, RunContext
from nnarith.utils import sanitize_label
from nnarith.visualization.terminal import print_matrix

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
            stem = f"{sanitize_label(arch_name)}__{sanitize_label(op_name)}"
            plot_path = self._current_run_dir / f"{stem}.png"
            print_matrix(
                matrix,
                cmap=self._config.colormap,
                precision=self._config.matrix_precision,
            )
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
