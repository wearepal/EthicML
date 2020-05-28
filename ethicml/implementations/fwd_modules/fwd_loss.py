from typing import Optional

from torch import nn
from torch.nn.modules.loss import _Loss

__all__ = ["FWDLoss"]


class FWDLoss(nn.Module):
    def __init__(self, loss_module: Optional[_Loss] = None, eta: float = 0.5):
        super().__init__()
        self.loss = loss_module
        self.eta = eta

    def forward(self, input, target):
        return (self.loss(input, target,) - self.eta).relu().square()
