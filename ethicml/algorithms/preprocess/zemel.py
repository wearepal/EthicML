"""
Zemel's Learned Fair Representations
"""
from typing import Dict, Union, Tuple, List

from ethicml.utility.data_structures import TestTuple, DataTuple, PathTuple, TestPathTuple
from .pre_algorithm import PreAlgorithmAsync
from .interface import flag_interface


class Zemel(PreAlgorithmAsync):
    """
    AIF360 implementation of Zemel's LFR
    """

    def __init__(
        self,
        threshold=0.5,
        clusters=2,
        Ax=0.01,
        Ay=0.1,
        Az=0.5,
        max_iter=5000,
        maxfun=5000,
        epsilon=1e-5,
    ):
        super().__init__()
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

    def run(self, train, test) -> Tuple[DataTuple, TestTuple]:
        from ...implementations import zemel

        return zemel.train_and_transform(train, test, self.flags)

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_paths: PathTuple,
        new_test_paths: TestPathTuple,
    ) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_paths, new_test_paths, self.flags)
        return ["-m", "ethicml.implementations.zemel"] + args

    @property
    def name(self) -> str:
        return "Zemel"
