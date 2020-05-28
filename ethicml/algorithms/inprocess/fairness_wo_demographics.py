from pathlib import Path
from typing import Dict, List, Optional, Union

from ethicml.algorithms.inprocess import InAlgorithmAsync
from ethicml.algorithms.inprocess.shared import flag_interface

__all__ = ["FWD"]


class FWD(InAlgorithmAsync):
    def __init__(
        self,
        eta: float,
        epochs: int = 10,
        batch_size: int = 32,
        network_size: Optional[List[int]] = None,
    ):
        super().__init__(name="FWD")
        if network_size is None:
            network_size = [50]
        self.flags: Dict[str, Union[int, str, List[int]]] = {
            "eta": eta,
            "batch_size": batch_size,
            "epochs": epochs,
            "network_size": network_size,
        }

    def _script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(train_path, test_path, pred_path, self.flags)
        return ["-m", "ethicml.implementations.fwd_tabular"] + args
