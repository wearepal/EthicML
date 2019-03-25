"""
Abstract Base Class of all algorithms in the framework
"""

from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, List, Any, Dict
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
        train_path = tmp_path / "transform_train.feather"
        test_path = tmp_path / "transform_test.feather"
        cmd = self._script_command(train_paths, test_paths, train_path, test_path)
        self._call_script(cmd)
        return train_path, test_path

    def _script_command(self, train_paths: PathTuple, test_paths: PathTuple, new_train_path: Path,
                        new_test_path: Path) -> List[str]:
        """The command that will run the script"""
        raise NotImplementedError("`_script_command` has not been implemented")

    @staticmethod
    def _flag_interface(train_paths: PathTuple, test_paths: PathTuple, new_train_path: Path,
                        new_test_path: Path, flags: Dict[str, Any]) -> List[str]:
        """Generate the commandline arguments that are expected"""
        flags_list: List[str] = []

        # paths to training and test data
        flags_list += Algorithm._path_tuple_to_cmd_args([train_paths, test_paths],
                                                        ['--train_', '--test_'])

        # paths to output files
        flags_list += ['--train_new', str(new_train_path), '--test_new', str(new_test_path)]

        # model parameters
        for key, values in flags.items():
            flags_list.append(f"--{key}")
            if isinstance(values, list):
                flags_list += [str(value) for value in values]
            else:
                flags_list.append(str(values))
        return flags_list
