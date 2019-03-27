"""
Abstract Base Class of all algorithms in the framework
"""
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd

from ethicml.algorithms.algorithm_base import Algorithm
from ethicml.algorithms.utils import (DataTuple, get_subset, PathTuple, write_as_feather,
                                      load_feather)


class InAlgorithm(Algorithm):
    """Abstract Base Class dor algorithms that run in the middle of the pipeline"""

    def run(self, train: DataTuple, test: DataTuple, sub_process: bool = False) -> pd.DataFrame:
        """Run Algorithm either in process or out of process

        Args:
            train: training data
            test: test data
            sub_process: indicate if the algorithm is to be run in it's own process
        """
        return self.run_threaded(train, test) if sub_process else self._run(train, test)

    @abstractmethod
    def _run(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        """Run Algorithm on the given data"""
        raise NotImplementedError("`_run` needs to be implemented")

    def run_test(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        """Run with reduced training set so that it finishes quicker"""
        train_testing = get_subset(train)
        return self.run(train_testing, test)

    def run_threaded(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        """ orchestrator for threaded """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_as_feather(train, test, tmp_path)
            pred_path = tmp_path / "predictions.feather"
            cmd = self._script_command(train_paths, test_paths, pred_path)
            self._call_script(cmd)
            return load_feather(pred_path)

    def _script_command(self, train_paths: PathTuple, test_paths: PathTuple, pred_path: Path) -> (
            List[str]):
        """The command that will run the script"""
        raise NotImplementedError("`_script_command` is  not implemented")
