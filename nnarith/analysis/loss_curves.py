from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from torch import nn

from nnarith.analysis.base import Analysis
from nnarith.config import ArchConfig
from nnarith.history import HistoryRecord, LossSeries, PlotRequest, history_to_series
from nnarith.records import ArtifactRecord, RunContext
from nnarith.utils import plot_filename


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
                    ", ".join(f"{key}={value}" for key, value in zip(self._config.group_by, group_key))
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


__all__ = ["LossCurveAnalysis", "LossCurveAnalysisConfig", "group_series_by_keys", "plot_series_group"]
