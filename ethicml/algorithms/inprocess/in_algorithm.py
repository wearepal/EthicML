"""Abstract Base Class of all algorithms in the framework."""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from abc import abstractmethod

from ethicml.common import implements
from ethicml.algorithms.algorithm_base import Algorithm, AlgorithmAsync, run_blocking
from ethicml.utility.data_structures import (
    DataTuple,
    TestTuple,
    PathTuple,
    TestPathTuple,
    Prediction,
    write_as_feather,
    load_prediction,
)


class InAlgorithm(Algorithm):
    """Abstract Base Class for algorithms that run in the middle of the pipeline."""

    def __init__(self, name: str, is_fairness_algo: bool = True):
        """Initialize the base class."""
        super().__init__(name=name)
        self.__is_fairness_algo = is_fairness_algo

    @abstractmethod
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data.

        Args:
            train: training data
            test: test data
        """

    def run_test(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run with reduced training set so that it finishes quicker."""
        train_testing = train.get_subset()
        return self.run(train_testing, test)

    @property
    def is_fairness_algo(self) -> bool:
        """True if this class corresponds to a fair algorithm."""
        return self.__is_fairness_algo


class InAlgorithmAsync(InAlgorithm, AlgorithmAsync):
    """In-Algorithm that can be run blocking and asynchronously."""

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run this asynchronous Algorithm as blocking on the given data.

        Args:
            train: training data
            test: test data
        """
        return run_blocking(self.run_async(train, test))

    async def run_async(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data asynchronously.

        Args:
            train: training data
            test: test data
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_as_feather(train, test, tmp_path)
            pred_path = tmp_path / "predictions.feather"
            cmd = self._script_command(train_paths, test_paths, pred_path)
            await self._call_script(cmd)  # wait for scrip to run
            return load_prediction(pred_path)

    @abstractmethod
    def _script_command(
        self, train_paths: PathTuple, test_paths: TestPathTuple, pred_path: Path
    ) -> (List[str]):
        """The command that will run the script."""
