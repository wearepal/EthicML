from typing import List

from torch import Tensor, nn
from torch.nn import BCELoss

from .fwd_loss import FWDLoss

__all__ = ["FWDClassifier"]


class FWDClassifier(nn.Module):
    def __init__(self, in_size: int, out_size: int, network_size: List[int], eta: float) -> None:
        super().__init__()

        self.loss = FWDLoss(loss_module=BCELoss(), eta=eta)

        self.seq = nn.Sequential()
        if not network_size:  # In the case that encoder size [] is specified
            self.seq.add_module("FWD Model single layer", nn.Linear(in_size, out_size))
        else:
            self.seq.add_module("FWD Model layer 0", nn.Linear(in_size, network_size[0]))
            self.seq.add_module("FWD Model activation 0", nn.ReLU())
            self.seq.add_module("FWD Model batch norm 0", nn.BatchNorm1d(network_size[0]))
            for k in range(len(network_size) - 1):
                self.seq.add_module(
                    f"FWD Model layer {k + 1}", nn.Linear(network_size[k], network_size[k + 1])
                )
                self.seq.add_module(f"FWD Model activation {k + 1}", nn.ReLU())
                self.seq.add_module(
                    f"FWD Model batch norm {k + 1}", nn.BatchNorm1d(network_size[k + 1])
                )
            self.seq.add_module("FWD Model last layer", nn.Linear(network_size[-1], out_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x).sigmoid()
