from __future__ import annotations

from nnarith.architecture import enumerate_architectures
from nnarith.config import DataSweep, SplitConfig, TrainingConfig
from nnarith.operations import add
from nnarith.presets import default_analysis_plan
from nnarith.training import run_experiments


def main() -> None:
    """Run a neural arithmetic experiment with explicit train/test ranges."""
    # Ranges that control both training data and downstream heatmap coverage.
    train_min, train_max = -20, 20
    heatmap_min, heatmap_max = -80, 80

    # Dataset configuration.
    base = 2
    train_samples = 50_000
    eval_samples = 10_000
    seed = 1234
    operations = (add,)

    train_split = SplitConfig(train_min, train_max, samples=train_samples)
    evaluation_splits = {
        "heatmap/test": SplitConfig(heatmap_min, heatmap_max, samples=eval_samples),
    }

    data_sweeps = {
        f"train [{train_min}, {train_max}]": DataSweep(
            base=base,
            train=train_split,
            evaluations=evaluation_splits,
            seed=seed,
            operations=operations,
        )
    }

    # Training configuration.
    training_configs = {
        "default_training": TrainingConfig(
            batch_size=512,
            epochs=10,
            learning_rate=1e-3,
        )
    }

    # Architecture grid.
    layer_options = [(64,)]
    dropouts = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.45, 0.5)
    l2_penalties = (0.0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)
    architectures = enumerate_architectures(layer_options, dropouts, l2_penalties)

    # Analyses and experiment execution.
    analysis_plan = default_analysis_plan(verbose=False, write_run_records=True)

    run_experiments(
        data_sweeps=data_sweeps,
        training_configs=training_configs,
        architectures=architectures,
        results_dir="results",
        analysis_plan=analysis_plan,
        torch_seed=0,
        write_run_records=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
