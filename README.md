## nnarith

Tools for building, training, and analysing small neural networks that learn
basic arithmetic operations.

### Key modules

- `nnarith.datasets`: encode integers into neural-friendly tensors and build
  reproducible train/eval datasets over configurable ranges.
- `nnarith.training`: run experiment sweeps over data splits, training configs,
  and architecture grids while collecting metrics and optional artefacts.
- `nnarith.analysis`: post-training analysis helpers (loss curves, heatmaps).
- `nnarith.presets`: batteries-included presets for common experiment setups.

### Quick start

Run the default experiment (single data sweep, training config, and a small
architecture grid) directly from the repository:

```bash
python main.py
```

The same behaviour is available programmatically:

```python
from nnarith.presets import run_default_experiment

run_default_experiment()
```

### Building your own scenario

Use the presets to start with a sensible baseline and then customise:

```python
from nnarith.presets import build_default_scenario, default_analysis_plan
from nnarith.training import run_experiments

# Start from the defaults and tweak any field you care about.
scenario = build_default_scenario(
    operand_limit=32,
    operations=[],  # supply your own tuple of callables if needed
)

analysis = default_analysis_plan(verbose=True)

run_experiments(
    data_sweeps=scenario.data_sweeps,
    training_configs=scenario.training_configs,
    architectures=scenario.architectures,
    results_dir="results",
    analysis_plan=analysis,
)
```

If you prefer a fully manual setup, construct the configurations directly:

```python
from nnarith.config import DataSweep, SplitConfig, TrainingConfig, ArchConfig
from nnarith.operations import add, subtract
from nnarith.training import run_experiments

data_sweeps = {
    "train [-10, 10]": DataSweep(
        base=10,
        train=SplitConfig(-10, 10, samples=5000),
        evaluations={
            "test": SplitConfig(-20, 20, samples=5000),
        },
        seed=1234,
        operations=(add, subtract),
    )
}

training = {
    "adam_default": TrainingConfig(batch_size=256, epochs=5, learning_rate=1e-3)
}

architectures = {
    "hidden64": ArchConfig(hidden_layers=(64,), dropout=0.1),
}

run_experiments(
    data_sweeps=data_sweeps,
    training_configs=training,
    architectures=architectures,
    results_dir="results",
)
```

### Dataset utilities

The dataset module lets you work with encoded numbers outside the training loop:

```python
from nnarith.datasets import generate_dataset_for_range
from nnarith.operations import add

dataset, encoding, *_ = generate_dataset_for_range(
    min_value=-10,
    max_value=10,
    operations=(add,),
    base=4,
    samples=1000,
    seed=42,
)

print(encoding.input_size, encoding.target_size)
```

This is useful for prototypes where you want direct access to the raw tensors.
