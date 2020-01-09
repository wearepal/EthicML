"""Simple upsampler that makes subgroups the same size as the majority group."""

from typing import List, Tuple, Dict

from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithmAsync
from ethicml.algorithms.preprocess.interface import flag_interface
from ethicml.utility import PathTuple, TestPathTuple, TestTuple, DataTuple


class Upsampler(PreAlgorithmAsync):
    """Upsampler algorithm.

    Given a datatuple, create a larger datatuple such that the subgroups have a balanced number
    of samples.
    """

    def __init__(self, strategy: str = "uniform"):
        super().__init__()

        assert strategy in ["uniform", "preferential", "naive"]
        self.strategy = strategy
        self.flags: Dict[str, str] = {"strategy": strategy}

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_paths: PathTuple,
        new_test_paths: TestPathTuple,
    ) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_paths, new_test_paths, self.flags)

        return ["-m", "ethicml.implementations.upsampler"] + args

    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        from ...implementations import upsampler

        return upsampler.train_and_transform(train, test, self.flags)

    @property
    def name(self) -> str:
        return f"Upsample {self.strategy}"
