"""
Abstract Base Class of all algorithms in the framework
"""

from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, List
import pandas as pd

from ethicml.algorithms.algorithm_base import Algorithm
from ..utils import get_subset, DataTuple, PathTuple, write_as_feather, load_feather


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
        return self.run_threaded(train, test) if sub_process else self._run(train, test)

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
            train_paths, test_paths = write_as_feather(train, test, tmp_path)
            train_path = tmp_path / "transform_train.feather"
            test_path = tmp_path / "transform_test.feather"
            cmd = self._script_command(train_paths, test_paths, train_path, test_path)
            self._call_script(cmd)
            return load_feather(train_path), load_feather(test_path)

    def _script_command(self, train_paths: PathTuple, test_paths: PathTuple, new_train_path: Path,
                        new_test_path: Path) -> List[str]:
        """The command that will run the script"""
        raise NotImplementedError("`_script_command` has not been implemented")
