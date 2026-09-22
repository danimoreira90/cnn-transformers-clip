"""Unit tests for the training loop shared by every notebook."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cv_mba.training import EpochRecord, predict, train_classifier


def separable_loader(n: int = 64, batch: int = 16) -> DataLoader:
    """Two clusters a linear model can separate in a handful of epochs."""
    torch.manual_seed(0)
    positive = torch.randn(n // 2, 4) + 3.0
    negative = torch.randn(n // 2, 4) - 3.0
    x = torch.cat([positive, negative])
    y = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)]).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True)


def test_one_record_per_epoch() -> None:
    model = nn.Linear(4, 2)
    loader = separable_loader()

    history = train_classifier(model, loader, loader, epochs=3,
                               optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    assert len(history) == 3
    assert [record.epoch for record in history] == [1, 2, 3]
    assert all(isinstance(record, EpochRecord) for record in history)


def test_every_record_carries_both_curves() -> None:
    # Rubric 1.2 asks for loss and accuracy curves per epoch. Both splits, every epoch.
    model = nn.Linear(4, 2)
    loader = separable_loader()

    record = train_classifier(model, loader, loader, epochs=1,
                              optimizer=torch.optim.SGD(model.parameters(), lr=0.1))[0]

    for value in (record.train_loss, record.train_accuracy,
                  record.val_loss, record.val_accuracy):
        assert isinstance(value, float)
    assert 0.0 <= record.train_accuracy <= 1.0


def test_a_separable_problem_actually_gets_learned() -> None:
    # A training loop that runs without reducing loss is a loop that is not training.
    model = nn.Linear(4, 2)
    loader = separable_loader()

    history = train_classifier(model, loader, loader, epochs=8,
                               optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    assert history[-1].train_loss < history[0].train_loss
    assert history[-1].val_accuracy > 0.9


def test_frozen_parameters_are_still_frozen_afterwards() -> None:
    # Rubric 1.1 is feature extraction with a frozen backbone. If the freeze leaks, the
    # notebook silently does full fine-tuning and answers a different rubric line.
    torch.manual_seed(0)
    backbone = nn.Linear(4, 4)
    head = nn.Linear(4, 2)
    model = nn.Sequential(backbone, head)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    before = backbone.weight.detach().clone()
    loader = separable_loader()

    train_classifier(model, loader, loader, epochs=5,
                     optimizer=torch.optim.SGD(
                         [p for p in model.parameters() if p.requires_grad], lr=0.1))

    assert torch.equal(backbone.weight, before)
    assert not torch.equal(head.weight, head.weight.new_zeros(head.weight.shape))


def test_predictions_line_up_with_their_labels() -> None:
    model = nn.Linear(4, 2)
    loader = DataLoader(TensorDataset(torch.randn(10, 4),
                                      torch.tensor([0, 1] * 5)), batch_size=4)

    y_true, y_pred = predict(model, loader)

    assert len(y_true) == len(y_pred) == 10
    assert set(y_true.tolist()) == {0, 1}


def test_prediction_order_follows_the_loader() -> None:
    # Shuffling here would silently scramble the confusion matrix.
    model = nn.Linear(4, 2)
    labels = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    loader = DataLoader(TensorDataset(torch.randn(8, 4), labels), batch_size=3)

    y_true, _ = predict(model, loader)

    assert y_true.tolist() == labels.tolist()


def test_predicting_leaves_the_model_in_evaluation_mode() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 2)).train()
    loader = DataLoader(TensorDataset(torch.randn(4, 4), torch.zeros(4).long()), batch_size=2)

    predict(model, loader)

    assert not model.training


def test_history_converts_to_a_frame_for_plotting() -> None:
    model = nn.Linear(4, 2)
    loader = separable_loader()

    history = train_classifier(model, loader, loader, epochs=2,
                               optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    frame = EpochRecord.to_frame(history)

    assert list(frame.columns) == ["epoch", "train_loss", "train_accuracy",
                                   "val_loss", "val_accuracy"]
    assert len(frame) == 2
