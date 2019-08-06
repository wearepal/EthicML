"""
Simple upsampler that makes subgroups the same size as the majority group
"""

from pathlib import Path
from typing import List, Tuple, Dict

from ethicml.algorithms.preprocess import PreAlgorithmAsync
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
        new_train_x_path: Path,
        new_train_s_path: Path,
        new_train_y_path: Path,
        new_train_name_path: Path,
        new_test_x_path: Path,
        new_test_s_path: Path,
        new_test_name_path: Path,
    ) -> List[str]:
        args = flag_interface(
            train_paths,
            test_paths,
            new_train_x_path,
            new_train_s_path,
            new_train_y_path,
            new_train_name_path,
            new_test_x_path,
            new_test_s_path,
            new_test_name_path,
            self.flags,
        )

        return ["-m", "ethicml.implementations.upsampler"] + args

    def run(self, train, test) -> Tuple[DataTuple, TestTuple]:
        from ...implementations import upsampler

        return upsampler.train_and_transform(train, test)

    @property
    def name(self) -> str:
        return "Upsample"
