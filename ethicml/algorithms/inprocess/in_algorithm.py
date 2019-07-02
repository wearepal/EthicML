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
    Predictions)


class InAlgorithm(Algorithm):
    """Abstract Base Class for algorithms that run in the middle of the pipeline"""

    @abstractmethod
    def run(self, train: DataTuple, test: TestTuple) -> Predictions:
        """Run Algorithm on the given data

        Args:
            train: training data
            test: test data
        """

    def run_test(self, train: DataTuple, test: TestTuple) -> Predictions:
        """Run with reduced training set so that it finishes quicker"""
        train_testing = get_subset(train)
        return self.run(train_testing, test)


class InAlgorithmAsync(InAlgorithm, AlgorithmAsync):
    """In-Algorithm that can be run blocking and asynchronously"""

    def run(self, train: DataTuple, test: TestTuple):
        return run_blocking(self.run_async(train, test))

    async def run_async(self, train: DataTuple, test: TestTuple) -> Predictions:
        """Run Algorithm on the given data asynchronously"""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_as_feather(train, test, tmp_path)
            soft_pred_path = tmp_path / "soft_predictions.feather"
            hard_pred_path = tmp_path / "hard_predictions.feather"
            cmd = self._script_command(train_paths, test_paths, soft_pred_path, hard_pred_path)
            await self._call_script(cmd)  # wait for scrip to run
            soft_labels = load_feather(soft_pred_path)
            hard_labels = load_feather(hard_pred_path)
            if hard_labels.columns[0] == "None":
                hard_labels = None

            return Predictions(soft=soft_labels, hard=hard_labels)

    @abstractmethod
    def _script_command(
        self, train_paths: PathTuple, test_paths: TestPathTuple,
        soft_pred_path: Path, hard_pred_path: Path
    ) -> (List[str]):
        """The command that will run the script"""
