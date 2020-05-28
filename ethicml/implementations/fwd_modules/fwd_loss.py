"""FWD Loss."""
from typing import Optional

from torch import nn
from torch.nn.modules.loss import _Loss

from ethicml.common import implements

__all__ = ["FWDLoss"]


class FWDLoss(nn.Module):
    """Fairness Without Demographics Loss."""

    def __init__(self, loss_module: Optional[_Loss] = None, eta: float = 0.5):
        """Set up the loss, set which loss you want to optimize and the eta to offset by."""
        super().__init__()
        self.loss = loss_module
        self.eta = eta

    @implements(nn.Module)
    def forward(self, input, target):
        return (self.loss(input, target,) - self.eta).relu().square()
