"""Utility Functins for HGR method."""
from __future__ import annotations

import torch


def compute_acc(yhat: torch.Tensor, y: torch.Tensor) -> float:
    """Accuracy."""
    _, predicted = torch.max(yhat, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return correct / total
