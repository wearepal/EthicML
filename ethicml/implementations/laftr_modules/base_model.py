"""Define a base model."""

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from ethicml import implements


class BaseModel(nn.Module):
    """Base model."""

    @implements(nn.Module)
    def __init__(self) -> None:
        super().__init__()

    @property
    def optim(self) -> torch.optim.Optimizer:
        """Optimiser for the model."""
        if self._optim is None:
            raise RuntimeError("Need to set the optimiser!")
        return self._optim

    @optim.setter
    def optim(self, optimiser: torch.optim.Optimizer) -> None:
        self._optim = optimiser

    @property
    def device(self) -> torch.device:
        """Device for the model."""
        if self._device is None:
            raise RuntimeError("Need to set a device!")
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self.to(device)
        self._device = device

    def log_results(self, results_dict: Dict[str, float]) -> None:
        """Log results."""
        epoch = results_dict.pop("epoch")
        epochs = results_dict.pop("epochs")
        print(f"{epoch} / {epochs}", {k: round(v, 4) for k, v in results_dict.items()})

    def unpack(self, j: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        """Unpack a dataloader's iterable."""
        x, s, _ = j
        return x.to(self.device), s.to(self.device)
