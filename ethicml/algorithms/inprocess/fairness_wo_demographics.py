"""Fairness without Demographics."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from ethicml.common import implements

from .in_algorithm import InAlgorithmAsync
from .shared import flag_interface

__all__ = ["DRO"]


class DRO(InAlgorithmAsync):
    """Implementation of https://arxiv.org/abs/1806.08010 ."""

    def __init__(
        self,
        eta: float = 0.5,
        epochs: int = 10,
        batch_size: int = 32,
        network_size: Optional[List[int]] = None,
    ):
        super().__init__(name="Dist Robust Optim")
        if network_size is None:
            network_size = [50]
        self.flags: Dict[str, Union[float, int, str, List[int]]] = {
            "eta": eta,
            "batch_size": batch_size,
            "epochs": epochs,
            "network_size": network_size,
        }

    @implements(InAlgorithmAsync)
    def _script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(train_path, test_path, pred_path, self.flags)
        return ["-m", "ethicml.implementations.dro_tabular"] + args
