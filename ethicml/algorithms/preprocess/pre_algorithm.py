"""
Abstract Base Class of all algorithms in the framework
"""

from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, List
import pandas as pd

from ethicml.algorithms.algorithm_base import Algorithm
from ..utils import get_subset, DataTuple, write_data_tuple, PathTuple


class PreAlgorithm(Algorithm):
    """Abstract Base Class for all algorithms that do pre-processing"""
    def run(self, train: DataTuple, test: DataTuple, sub_process: bool = False) -> (
            Tuple[pd.DataFrame, pd.DataFrame]):
        """Generate fair features by either running in process or out of process

        Args:
            train: training data
            test: test data
            sub_process: should this model run in it's own process?
        """
        if sub_process:
            return self.run_threaded(train, test)
        return self._run(train, test)

    @abstractmethod
    def _run(self, train: DataTuple, test: DataTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate fair features with the given data"""
        raise NotImplementedError("`_run` needs to be implemented")

    def run_test(self, train: DataTuple, test: DataTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run with reduced training set so that it finishes quicker"""
        train_testing = get_subset(train)
        return self.run(train_testing, test)

    def run_threaded(self, train: DataTuple, test: DataTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ orchestrator for threaded """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_data_tuple(train, test, tmp_path)
            train_path, test_path = self.run_thread(train_paths, test_paths, tmp_path)
            return self._load_output(train_path), self._load_output(test_path)

    def run_thread(self, train_paths, test_paths, tmp_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run algorithm in its own thread"""
        train_path = tmp_path / "transform_train.parquet"
        test_path = tmp_path / "transform_test.parquet"
        args = self._script_interface(train_paths, test_paths, train_path, test_path)
        self._call_script(['-m', self.__module__] + args)
        return train_path, test_path

    @staticmethod
    def save_transformations(transforms: Tuple[pd.DataFrame, pd.DataFrame], flags):
        """Save the data to the file that was specified in the commandline arguments"""
        transform_train, transform_test = transforms
        assert isinstance(transform_train, pd.DataFrame)
        assert isinstance(transform_test, pd.DataFrame)
        transform_train_path = Path(flags.train_new)
        transform_train.columns = transform_train.columns.astype(str)
        transform_train.to_parquet(transform_train_path, compression=None)
        transform_test_path = Path(flags.test_new)
        transform_test.columns = transform_test.columns.astype(str)
        transform_test.to_parquet(transform_test_path, compression=None)

    @abstractmethod
    def _script_interface(self, train_paths: PathTuple, test_paths: PathTuple, new_train_path: Path,
                          new_test_path: Path) -> List[str]:
        """Generate the commandline arguments that are expected by the script"""


class PreAlgorithmCommon(PreAlgorithm):
    """Common functionality that many pre-algorithms will share"""
    # pylint: disable=abstract-method
    def __init__(self, flags, **kwargs):
        super().__init__(**kwargs)
        self.flags = flags

    def _script_interface(self, train_paths, test_paths, new_train_path, new_test_path):
        """Generate the commandline arguments that are expected by the Beutel script"""
        flags_list: List[str] = []

        # paths to training and test data
        flags_list += self._path_tuple_to_cmd_args([train_paths, test_paths],
                                                   ['--train_', '--test_'])

        # paths to output files
        flags_list += ['--train_new', str(new_train_path), '--test_new', str(new_test_path)]

        # model parameters
        for key, values in self.flags.items():
            flags_list.append(f"--{key}")
            flags_list += values
        return flags_list
