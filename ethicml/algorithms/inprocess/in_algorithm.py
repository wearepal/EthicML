"""
Abstract Base Class of all algorithms in the framework
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from abc import abstractmethod

import pandas as pd

from ethicml.algorithms.algorithm_base import Algorithm, AlgorithmAsync, run_blocking
from ethicml.utility.data_structures import (
    DataTuple,
    TestTuple,
    get_subset,
    PathTuple,
    TestPathTuple,
    write_as_feather,
    load_feather,
)


class InAlgorithm(Algorithm):
    """Abstract Base Class for algorithms that run in the middle of the pipeline"""

    @abstractmethod
    def run(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        """Run Algorithm on the given data

        Args:
            train: training data
            test: test data
        """

    def run_test(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        """Run with reduced training set so that it finishes quicker"""
        train_testing = get_subset(train)
        return self.run(train_testing, test)


class InAlgorithmAsync(InAlgorithm, AlgorithmAsync):
    """In-Algorithm that can be run blocking and asynchronously"""

    def run(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        return run_blocking(self.run_async(train, test))

    async def run_async(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        """Run Algorithm on the given data asynchronously"""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_as_feather(train, test, tmp_path)
            pred_path = tmp_path / "predictions.feather"
            cmd = self._script_command(train_paths, test_paths, pred_path)
            await self._call_script(cmd)  # wait for scrip to run
            return load_feather(pred_path)

    @abstractmethod
    def _script_command(
        self, train_paths: PathTuple, test_paths: TestPathTuple, pred_path: Path
    ) -> (List[str]):
        """The command that will run the script"""
