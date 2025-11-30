from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gen import generate_arithmetic_datasets


def add(left: int, right: int) -> int:
    return left + right

def subtract(left: int, right: int) -> int:
    return left - right

def multiply(left: int, right: int) -> int:
    return left * right


@dataclass(frozen=True)
class DataConfig:
    train_min: int
    train_max: int
    test_min: int
    test_max: int
    base: int
    train_samples: int
    test_samples: int
    seed: int
    operations: Tuple[Callable[[int, int], int], ...]


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    device: Optional[torch.device] = None

lim = 99

default = DataConfig(
        train_min=-lim,
        train_max=lim,
        test_min=-lim,
        test_max=lim,
        base=4,
        train_samples=50_000,
        test_samples=10_000,
        seed=1234,
        operations=(add, subtract, multiply),
    )

DATA_CONFIGS: Dict[str, DataConfig] = {
    f"test max = {ceil}x train": replace(default, test_max=int(lim*ceil)) for ceil in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
}

TRAINING_CONFIGS: Dict[str, TrainingConfig] = {
    "default_training": TrainingConfig(
        batch_size=512,
        epochs=20,
        learning_rate=1e-3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ),
}

PLOT_BASENAME = "loss_curves"


def resolve_device(training_config: TrainingConfig) -> torch.device:
    device = training_config.device
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_filename(base: str, data_name: str, training_name: str) -> str:
    def sanitize(label: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in label)

    return f"{base}_{sanitize(data_name)}__{sanitize(training_name)}.png"


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: Sequence[int],
    activation: Callable[[], nn.Module] = nn.ReLU,
) -> nn.Module:
    layers: List[nn.Module] = []
    previous = input_dim
    for width in hidden_layers:
        layers.append(nn.Linear(previous, width))
        layers.append(activation())
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


arches = (
    [64],
    [64, 64],
    [128],
    [128, 128],
    # [256],
    # [256, 256],
    # [256, 256, 256],
    # [512],
    # [512, 512],
    # [1024, 512],
)

ARCHITECTURES: Dict[str, Callable[[int, int], nn.Module]] = {
    str(layers): lambda in_dim, out_dim: build_mlp(in_dim, out_dim, layers) for layers in arches
}


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


def main() -> None:
    torch.manual_seed(0)

    plots_to_save: List[Tuple[str, str, str, Dict[str, Dict[str, List[float]]]]] = []
    global_epoch_max = 0
    global_loss_min = float("inf")
    global_loss_max = float("-inf")

    for data_name, data_config in DATA_CONFIGS.items():
        datasets = generate_arithmetic_datasets(
            data_config.train_min,
            data_config.train_max,
            data_config.test_min,
            data_config.test_max,
            data_config.operations,
            data_config.base,
            train_samples=data_config.train_samples,
            test_samples=data_config.test_samples,
            seed=data_config.seed,
        )

        print(f"\n=== Data config: {data_name} ===")
        print(f"Input size: {datasets.input_size}, Target size: {datasets.target_size}")

        for training_name, training_config in TRAINING_CONFIGS.items():
            device = resolve_device(training_config)

            print(f"\n--- Training config: {training_name} (device={device}) ---")

            train_loader = DataLoader(
                datasets.train,
                batch_size=training_config.batch_size,
                shuffle=True,
            )
            test_loader = DataLoader(
                datasets.test,
                batch_size=training_config.batch_size,
                shuffle=False,
            )

            criterion = nn.MSELoss()
            results = []
            histories: Dict[str, Dict[str, List[float]]] = {}

            for arch_name, builder in ARCHITECTURES.items():
                model = builder(datasets.input_size, datasets.target_size).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=training_config.learning_rate
                )
                history = {"train": [], "test": []}
                for epoch in range(training_config.epochs):
                    train_loss = train_epoch(
                        model, train_loader, optimizer, criterion, device
                    )
                    test_loss = evaluate(model, test_loader, criterion, device)
                    history["train"].append(train_loss)
                    history["test"].append(test_loss)
                    print(
                        f"{arch_name} ({data_name} | {training_name}): "
                        f"epoch={epoch + 1} train_loss={train_loss:.6f} "
                        f"test_loss={test_loss:.6f}"
                    )
                histories[arch_name] = history
                results.append((arch_name, history["train"][-1], history["test"][-1]))

            if results:
                ranked = sorted(results, key=lambda item: item[2])
                print(f"\nRanked by test loss ({data_name} | {training_name}):")
                for rank, (name, train_loss, test_loss) in enumerate(
                    ranked, start=1
                ):
                    print(
                        f"{rank}. {name}: train_loss={train_loss:.6f}, "
                        f"test_loss={test_loss:.6f}"
                    )

            if histories:
                path = plot_filename(PLOT_BASENAME, data_name, training_name)
                plots_to_save.append((path, data_name, training_name, histories))

                for history in histories.values():
                    epochs = len(history["train"])
                    global_epoch_max = max(global_epoch_max, epochs)
                    combined_losses = history["train"] + history["test"]
                    if combined_losses:
                        global_loss_min = min(global_loss_min, min(combined_losses))
                        global_loss_max = max(global_loss_max, max(combined_losses))

    if not plots_to_save:
        return

    global_epoch_max = max(global_epoch_max, 1)
    x_limits = (1, global_epoch_max)

    if global_loss_min == float("inf"):
        global_loss_min = 0.0
    if global_loss_max == float("-inf"):
        global_loss_max = 1.0

    loss_span = max(global_loss_max - global_loss_min, 1e-8)
    margin = loss_span * 0.05
    y_limits = (global_loss_min - margin, global_loss_max + margin)

    for path, data_name, training_name, histories in plots_to_save:
        plot_histories(histories, path, data_name, training_name, x_limits, y_limits)
        print(f"Saved loss curves to {path}")


def plot_histories(
    histories: Dict[str, Dict[str, List[float]]],
    path: str,
    data_name: str,
    training_name: str,
    x_limits: Tuple[int, int],
    y_limits: Tuple[float, float],
) -> None:
    plt.figure(figsize=(8, 5))
    color_cycle = plt.cm.tab10.colors
    for idx, (name, history) in enumerate(histories.items()):
        color = color_cycle[idx % len(color_cycle)]
        epochs = range(1, len(history["train"]) + 1)
        plt.plot(epochs, history["train"], color=color, linewidth=1.8, label=f"{name} train")
        plt.plot(epochs, history["test"], color=color, linestyle="--", linewidth=1.5, label=f"{name} test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training vs Test Loss\n(data={data_name}, training={training_name})")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.xlim(*x_limits)
    plt.ylim(*y_limits)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
