"""DRO Loss."""

from typing import Type
from typing_extensions import override

from torch import Tensor, nn
from torch.nn.modules.loss import NLLLoss, _Loss

__all__ = ["DROLoss"]


class DROLoss(nn.Module):
    """Fairness Without Demographics Loss."""

    def __init__(self, loss_module: Type[_Loss] | None = None, eta: float = 0.5):
        super().__init__()
        if loss_module is None:
            loss_module = NLLLoss
        self.loss = loss_module(reduction="none")
        self.eta = eta

    @override
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return (self.loss(pred, target=target) - self.eta).relu().pow(2).mean()
