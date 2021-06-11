"""Abstract Base Class of all algorithms in the framework."""
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

from kit import implements

from ethicml.algorithms.algorithm_base import Algorithm, AlgorithmAsync, run_blocking
from ethicml.utility import DataTuple, Prediction, TestTuple

__all__ = ["InAlgorithm", "InAlgorithmAsync"]


class InAlgorithm(Algorithm):
    """Abstract Base Class for algorithms that run in the middle of the pipeline."""

    def __init__(self, name: str, is_fairness_algo: bool = True):
        super().__init__(name=name)
        self.__is_fairness_algo = is_fairness_algo

    @abstractmethod
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data.

        Args:
            train: training data
            test: test data

        Returns:
            predictions
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

        Returns:
            predictions
        """
        return run_blocking(self.run_async(train, test))

    async def run_async(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data asynchronously.

        Args:
            train: training data
            test: test data

        Returns:
            predictions
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = tmp_path / "train.npz"
            test_path = tmp_path / "test.npz"
            pred_path = tmp_path / "predictions.npz"
            train.to_npz(train_path)
            test.to_npz(test_path)
            cmd = self._script_command(train_path, test_path, pred_path)
            await self._call_script(cmd)  # wait for scrip to run
            return Prediction.from_npz(pred_path)

    @abstractmethod
    def _script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        """The command that will run the script."""
