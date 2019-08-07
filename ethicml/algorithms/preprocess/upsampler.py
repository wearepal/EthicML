"""
Simple upsampler that makes subgroups the same size as the majority group
"""

from typing import List, Tuple, Dict

from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithmAsync
from ethicml.algorithms.preprocess.interface import flag_interface
from ethicml.utility import PathTuple, TestPathTuple, TestTuple, DataTuple


class Upsampler(PreAlgorithmAsync):
    """
    Upsampler class.
    Given a datatuple, create a larger datatuple such that the subgroups have a balanced number
    of samples.
    """

    def __init__(self):
        super().__init__()
        self.flags: Dict[str, str] = {}

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_paths: PathTuple,
        new_test_paths: TestPathTuple,
    ) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_paths, new_test_paths, self.flags)

        return ["-m", "ethicml.implementations.upsampler"] + args

    def run(self, train, test) -> Tuple[DataTuple, TestTuple]:
        from ...implementations import upsampler

        return upsampler.train_and_transform(train, test)

    @property
    def name(self) -> str:
        return "Upsample"
