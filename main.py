from typing import Dict, Tuple

from train_models import (
    ArchConfig,
    AnalysisPlan,
    DataSweep,
    HeatmapAnalysisConfig,
    LossCurveAnalysisConfig,
    SplitConfig,
    TrainingConfig,
    add,
    enumerate_architectures,
    run_experiments,
)


def build_scenario() -> Tuple[
    Dict[str, DataSweep],
    Dict[str, TrainingConfig],
    Dict[str, ArchConfig],
]:
    lim = 20
    train_split = SplitConfig(minimum=-lim, maximum=lim, samples=50_000)
    evaluation_splits: Dict[str, SplitConfig] = {
        f"test max = {ceil}x train": SplitConfig(
            minimum=-int(lim * ceil),
            maximum=int(lim * ceil),
            samples=10_000,
        )
        for ceil in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    }

    data_sweeps: Dict[str, DataSweep] = {
        "train [-20, 20]": DataSweep(
            base=4,
            train=train_split,
            evaluations=evaluation_splits,
            seed=1234,
            operations=(add,),
        )
    }

    training_configs: Dict[str, TrainingConfig] = {
        "default_training": TrainingConfig(
            batch_size=512,
            epochs=10,
            learning_rate=1e-3,
        ),
    }

    layer_options = [
        (64,),
    ]
    dropouts = (0.0, 0.3)
    l2_penalties = (0.0, 2e-5)
    architectures = enumerate_architectures(layer_options, dropouts, l2_penalties)

    return data_sweeps, training_configs, architectures


def main() -> None:
    data_sweeps, training_configs, architectures = build_scenario()
    analysis_plan = AnalysisPlan(
        loss_curves=LossCurveAnalysisConfig(
            basename="loss_curves",
            group_by=("split",),
            base_selectors=({"series_kind": "train"},),
        ),
        network_heatmaps=HeatmapAnalysisConfig(basename="heatmaps"),
    )
    run_experiments(
        data_sweeps=data_sweeps,
        training_configs=training_configs,
        architectures=architectures,
        results_dir="results",
        analysis_plan=analysis_plan,
        torch_seed=0,
    )


if __name__ == "__main__":
    main()
