"""
Abstract Base Class of all algorithms in the framework
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from abc import abstractmethod

import pandas as pd

from ethicml.algorithms.algorithm_base import Algorithm, AlgorithmAsync
from ethicml.algorithms.utils import (
    DataTuple,
    get_subset,
    PathTuple,
    write_as_feather,
    load_feather,
)


class InAlgorithmSync(Algorithm):
    """Abstract Base Class for algorithms that run in the middle of the pipeline"""

    @abstractmethod
    def run(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        """Run Algorithm on the given data

        Args:
            train: training data
            test: test data
        """

    def run_test(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        """Run with reduced training set so that it finishes quicker"""
        train_testing = get_subset(train)
        return self.run(train_testing, test)


class InAlgorithmAsync(AlgorithmAsync):
    """Abstract Base Class for algorithms that run asynchronously in the middle of the pipeline"""

    async def run_async(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        """Run Algorithm on the given data asynchronously"""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_as_feather(train, test, tmp_path)
            pred_path = tmp_path / "predictions.feather"
            cmd = self._script_command(train_paths, test_paths, pred_path)
            self._call_script(cmd)
            return load_feather(pred_path)

    @abstractmethod
    def _script_command(
        self, train_paths: PathTuple, test_paths: PathTuple, pred_path: Path
    ) -> (List[str]):
        """The command that will run the script"""


class InAlgorithm(InAlgorithmSync, InAlgorithmAsync):
    """In-Algorithm that can be run blocking and asynchronously"""

    @abstractmethod
    def run(self, train, test):
        pass
