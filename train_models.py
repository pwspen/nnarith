from typing import Callable, Dict, List, Sequence

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

DATA_CONFIG = {
    "train_min": -99,
    "train_max": 99,
    "test_min": -99,
    "test_max": 199,
    "base": 4,
    "train_samples": 50_000,
    "test_samples": 10_000,
    "seed": 1234,
    "operations": (add, subtract, multiply),
}

TRAINING_CONFIG = {
    "batch_size": 512,
    "epochs": 20,
    "learning_rate": 1e-3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

PLOT_PATH = "loss_curves.png"


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


ARCHITECTURES: Dict[str, Callable[[int, int], nn.Module]] = {
    "mlp_small": lambda in_dim, out_dim: build_mlp(in_dim, out_dim, [256]),
    "mlp_medium": lambda in_dim, out_dim: build_mlp(in_dim, out_dim, [512, 256]),
    "mlp_wide": lambda in_dim, out_dim: build_mlp(in_dim, out_dim, [1024, 512]),
    "mlp_deep": lambda in_dim, out_dim: build_mlp(in_dim, out_dim, [512] * 4),
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
    device: torch.device = TRAINING_CONFIG["device"]

    datasets = generate_arithmetic_datasets(
        DATA_CONFIG["train_min"],
        DATA_CONFIG["train_max"],
        DATA_CONFIG["test_min"],
        DATA_CONFIG["test_max"],
        DATA_CONFIG["operations"],
        DATA_CONFIG["base"],
        train_samples=DATA_CONFIG["train_samples"],
        test_samples=DATA_CONFIG["test_samples"],
        seed=DATA_CONFIG["seed"],
    )

    print(f"Input size: {datasets.input_size}, Target size: {datasets.target_size}")

    train_loader = DataLoader(
        datasets.train,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.test,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
    )

    criterion = nn.MSELoss()
    results = []
    histories: Dict[str, Dict[str, List[float]]] = {}

    for name, builder in ARCHITECTURES.items():
        model = builder(datasets.input_size, datasets.target_size).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=TRAINING_CONFIG["learning_rate"]
        )
        history = {"train": [], "test": []}
        for epoch in range(TRAINING_CONFIG["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss = evaluate(model, test_loader, criterion, device)
            history["train"].append(train_loss)
            history["test"].append(test_loss)
            print(
                f"{name}: epoch={epoch + 1} train_loss={train_loss:.6f} "
                f"test_loss={test_loss:.6f}"
            )
        histories[name] = history
        results.append((name, history["train"][-1], history["test"][-1]))

    if results:
        ranked = sorted(results, key=lambda item: item[2])
        print("\nRanked by test loss:")
        for rank, (name, train_loss, test_loss) in enumerate(ranked, start=1):
            print(
                f"{rank}. {name}: train_loss={train_loss:.6f}, "
                f"test_loss={test_loss:.6f}"
            )

    if histories:
        plot_histories(histories, PLOT_PATH)
        print(f"\nSaved loss curves to {PLOT_PATH}")


def plot_histories(histories: Dict[str, Dict[str, List[float]]], path: str) -> None:
    plt.figure(figsize=(8, 5))
    color_cycle = plt.cm.tab10.colors
    for idx, (name, history) in enumerate(histories.items()):
        color = color_cycle[idx % len(color_cycle)]
        epochs = range(1, len(history["train"]) + 1)
        plt.plot(epochs, history["train"], color=color, linewidth=1.8, label=f"{name} train")
        plt.plot(epochs, history["test"], color=color, linestyle="--", linewidth=1.5, label=f"{name} test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
