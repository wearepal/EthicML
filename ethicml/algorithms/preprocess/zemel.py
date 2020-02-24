"""Zemel's Learned Fair Representations."""
from typing import Dict, List, Union

from ethicml.utility import PathTuple, TestPathTuple

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
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_paths: PathTuple,
        new_test_paths: TestPathTuple,
    ) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_paths, new_test_paths, self.flags)
        return ["-m", "ethicml.implementations.zemel"] + args
