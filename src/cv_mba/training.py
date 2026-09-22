"""The training loop every notebook shares.

Deliberately plain: one optimiser step per batch, one validation pass per epoch, one
record of four numbers kept for each. No scheduler, no early stopping, no mixed
precision. Activity 3 asks for a single training run with documented curves, and
Activity 1 compares two models that must differ only in what is being compared.
Machinery that changes the effective learning rate between runs would undermine both.

The loop is here rather than in a notebook so it can be tested. The test that matters is
`test_frozen_parameters_are_still_frozen_afterwards`: if a freeze leaks, the notebook
quietly performs full fine-tuning and answers a different rubric line than the one it
claims.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochRecord:
    """What one epoch produced, on both splits."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float

    @staticmethod
    def to_frame(history: list["EpochRecord"]) -> pd.DataFrame:
        """The history as a table, ready to plot or drop into the report."""
        return pd.DataFrame(
            [astuple(record) for record in history],
            columns=[field.name for field in fields(EpochRecord)],
        )


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
    criterion: nn.Module | None = None,
    on_epoch=None,
) -> list[EpochRecord]:
    """Train for `epochs`, returning one record per epoch.

    `optimizer` is built by the caller, which is what keeps feature extraction honest:
    the caller decides which parameters it is given, and a frozen backbone stays frozen
    because its parameters were never handed over.
    """
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    model.to(device)
    history: list[EpochRecord] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_seen += labels.size(0)

        val_loss, val_correct, val_seen = _evaluate(model, val_loader, criterion, device)

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss / train_seen,
            train_accuracy=train_correct / train_seen,
            val_loss=val_loss / val_seen,
            val_accuracy=val_correct / val_seen,
        )
        history.append(record)
        if on_epoch is not None:
            on_epoch(record)

    return history


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor]:
    """Return true and predicted labels, in the order the loader produced them.

    Order matters: these two go straight into a confusion matrix, and a shuffled
    validation loader would scramble it without any error appearing anywhere.
    """
    model.to(device).eval()
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predicted_labels.append(logits.argmax(dim=1).cpu())
            true_labels.append(labels)

    return torch.cat(true_labels), torch.cat(predicted_labels)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
) -> tuple[float, int, int]:
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss, correct, seen
