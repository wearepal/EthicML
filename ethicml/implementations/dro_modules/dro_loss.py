"""DRO Loss."""
from __future__ import annotations
from typing import Type

from ranzen import implements
from torch import Tensor, nn
from torch.nn.modules.loss import NLLLoss, _Loss

__all__ = ["DROLoss"]


class DROLoss(nn.Module):
    """Fairness Without Demographics Loss."""

    def __init__(self, loss_module: Type[_Loss] | None = None, eta: float = 0.5):
        """Set up the loss, set which loss you want to optimize and the eta to offset by."""
        super().__init__()
        if loss_module is None:
            loss_module = NLLLoss
        self.loss = loss_module(reduction="none")
        self.eta = eta

    @implements(nn.Module)
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return (self.loss(pred, target=target) - self.eta).relu().pow(2).mean()
