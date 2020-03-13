"""Zemel's Learned Fair Representations."""
from pathlib import Path
from typing import Dict, List, Union

from .interface import flag_interface
from .pre_algorithm import PreAlgorithmAsync


class Zemel(PreAlgorithmAsync):
    """AIF360 implementation of Zemel's LFR."""

    def __init__(
        self,
        threshold: float = 0.5,
        clusters: int = 2,
        Ax: float = 0.01,
        Ay: float = 0.1,
        Az: float = 0.5,
        max_iter: int = 5000,
        maxfun: int = 5000,
        epsilon: float = 1e-5,
    ) -> None:
        """Init Zemel."""
        super().__init__(name="Zemel")
        self.flags: Dict[str, Union[int, float]] = {
            "clusters": clusters,
            "Ax": Ax,
            "Ay": Ay,
            "Az": Az,
            "max_iter": max_iter,
            "maxfun": maxfun,
            "epsilon": epsilon,
            "threshold": threshold,
        }

    def _script_command(
        self, train_path: Path, test_path: Path, new_train_path: Path, new_test_path: Path
    ) -> List[str]:
        args = flag_interface(train_path, test_path, new_train_path, new_test_path, self.flags)
        return ["-m", "ethicml.implementations.zemel"] + args
