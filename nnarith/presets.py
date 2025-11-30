from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from nnarith.analysis.heatmaps import HeatmapAnalysisConfig
from nnarith.analysis.loss_curves import LossCurveAnalysisConfig
from nnarith.analysis.planning import AnalysisPlan
from nnarith.architecture import enumerate_architectures
from nnarith.config import ArchConfig, DataSweep, SplitConfig, TrainingConfig
from nnarith.operations import add
from nnarith.training import run_experiments

Operation = Callable[[int, int], int]


@dataclass(frozen=True)
class Scenario:
    data_sweeps: Dict[str, DataSweep]
    training_configs: Dict[str, TrainingConfig]
    architectures: Dict[str, ArchConfig]


_DEFAULT_LAYER_OPTIONS: Tuple[Tuple[int, ...], ...] = ((64,),)
_DEFAULT_DROPOUTS: Tuple[float, ...] = (0.0, 0.3)
_DEFAULT_L2: Tuple[float, ...] = (0.0, 2e-5)
_DEFAULT_EVALUATION_SCALES: Tuple[float, ...] = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)


def build_default_scenario(
    *,
    operand_limit: int = 20,
    base: int = 4,
    train_samples: int = 50_000,
    eval_samples: int = 10_000,
    evaluation_scales: Sequence[float] = _DEFAULT_EVALUATION_SCALES,
    operations: Optional[Sequence[Operation]] = None,
    seed: int = 1234,
    layer_options: Sequence[Sequence[int]] = _DEFAULT_LAYER_OPTIONS,
    dropouts: Sequence[float] = _DEFAULT_DROPOUTS,
    l2_penalties: Sequence[float] = _DEFAULT_L2,
) -> Scenario:
    ops: Tuple[Operation, ...] = tuple(operations or (add,))
    if not ops:
        raise ValueError("operations must not be empty")

    train_split = SplitConfig(minimum=-operand_limit, maximum=operand_limit, samples=train_samples)
    evaluation_splits: Dict[str, SplitConfig] = {
        f"test max = {scale}x train": SplitConfig(
            minimum=-int(operand_limit * scale),
            maximum=int(operand_limit * scale),
            samples=eval_samples,
        )
        for scale in evaluation_scales
    }

    data_sweeps = {
        f"train [-{operand_limit}, {operand_limit}]": DataSweep(
            base=base,
            train=train_split,
            evaluations=evaluation_splits,
            seed=seed,
            operations=ops,
        )
    }

    training_configs = {
        "default_training": TrainingConfig(
            batch_size=512,
            epochs=10,
            learning_rate=1e-3,
        )
    }

    architectures = enumerate_architectures(layer_options, dropouts, l2_penalties)

    return Scenario(
        data_sweeps=data_sweeps,
        training_configs=training_configs,
        architectures=architectures,
    )


def default_analysis_plan(
    *,
    verbose: bool = False,
    write_run_records: bool = True,
    loss_curves_basename: str = "loss_curves",
    heatmaps_basename: str = "heatmaps",
) -> AnalysisPlan:
    return AnalysisPlan(
        loss_curves=LossCurveAnalysisConfig(
            basename=loss_curves_basename,
            group_by=("split",),
            base_selectors=({"series_kind": "train"},),
            announce_paths=verbose,
        ),
        network_heatmaps=HeatmapAnalysisConfig(
            basename=heatmaps_basename,
            store_raw_matrix=write_run_records,
            preview_in_terminal=verbose,
        ),
    )


def run_default_experiment(
    *,
    results_dir: str = "results",
    torch_seed: Optional[int] = 0,
    write_run_records: bool = True,
    verbose: bool = False,
    scenario: Optional[Scenario] = None,
    scenario_overrides: Optional[Mapping[str, object]] = None,
    analysis_plan: Optional[AnalysisPlan] = None,
) -> None:
    if scenario is None:
        overrides = dict(scenario_overrides or {})
        scenario = build_default_scenario(**overrides)
    if analysis_plan is None:
        analysis_plan = default_analysis_plan(
            verbose=verbose,
            write_run_records=write_run_records,
        )

    run_experiments(
        data_sweeps=scenario.data_sweeps,
        training_configs=scenario.training_configs,
        architectures=scenario.architectures,
        results_dir=results_dir,
        analysis_plan=analysis_plan,
        torch_seed=torch_seed,
        write_run_records=write_run_records,
        verbose=verbose,
    )


__all__ = [
    "Scenario",
    "build_default_scenario",
    "default_analysis_plan",
    "run_default_experiment",
]
