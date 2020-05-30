"""DRO Loss."""
from typing import Optional, Type

from torch import Tensor, nn
from torch.nn.modules.loss import NLLLoss, _Loss

from ethicml.common import implements

__all__ = ["DROLoss"]


class DROLoss(nn.Module):
    """Fairness Without Demographics Loss."""

    def __init__(self, loss_module: Optional[Type[_Loss]] = None, eta: float = 0.5):
        """Set up the loss, set which loss you want to optimize and the eta to offset by."""
        super().__init__()
        if loss_module is None:
            loss_module = NLLLoss
        self.loss = loss_module(reduction="none")
        self.eta = eta

    @implements(nn.Module)
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return (self.loss(pred, target) - self.eta).relu().pow(2).mean()
