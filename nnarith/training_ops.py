from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from nnarith.config import TrainingConfig


def resolve_device(training_config: TrainingConfig) -> torch.device:
    device = training_config.device
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
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


__all__ = ["evaluate", "resolve_device", "train_epoch"]
