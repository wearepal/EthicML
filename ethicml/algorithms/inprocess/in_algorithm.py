"""Abstract Base Class of all algorithms in the framework."""
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, TypeVar

from ranzen import implements

from ethicml.algorithms.algorithm_base import Algorithm, SubprocessAlgorithmMixin
from ethicml.utility import DataTuple, Prediction, TestTuple

__all__ = ["InAlgorithm", "InAlgorithmAsync"]

_I = TypeVar("_I", bound="InAlgorithm")


class InAlgorithm(Algorithm):
    """Abstract Base Class for algorithms that run in the middle of the pipeline."""

    def __init__(self, name: str, seed: int, is_fairness_algo: bool = True):
        super().__init__(name=name, seed=seed)
        self.__is_fairness_algo = is_fairness_algo

    @abstractmethod
    def fit(self: _I, train: DataTuple) -> _I:
        """Run Algorithm on the given data.

        Args:
            train: training data

        Returns:
            self, but trained.
        """

    @abstractmethod
    def predict(self, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data.

        Args:
            test: data to evaluate on

        Returns:
            predictions
        """

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


_IA = TypeVar("_IA", bound="InAlgorithmAsync")


class InAlgorithmAsync(SubprocessAlgorithmMixin, InAlgorithm):
    """In-Algorithm that uses a subprocess to run."""

    @implements(InAlgorithm)
    def fit(self: _IA, train: DataTuple) -> _IA:
        """Fit algorithm on the given data asynchronously.

        Args:
            train: training data
            test: test data

        Returns:
            predictions
        """
        self.model_path = self.model_dir / f"model_{self.name}.joblib"
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = tmp_path / "train.npz"
            train.to_npz(train_path)
            cmd = self._fit_script_command(train_path, self.model_path)
            self._call_script(cmd + ["--mode", "fit"])  # wait for script to run
            return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data asynchronously.

        Args:
            train: training data
            test: test data

        Returns:
            predictions
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_path = tmp_path / "test.npz"
            pred_path = tmp_path / "predictions.npz"
            test.to_npz(test_path)
            cmd = self._predict_script_command(self.model_path, test_path, pred_path)
            self._call_script(cmd + ["--mode", "predict"])  # wait for scrip to run
            return Prediction.from_npz(pred_path)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
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
            cmd = self._run_script_command(train_path, test_path, pred_path)
            self._call_script(cmd + ["--mode", "run"])  # wait for scrip to run
            return Prediction.from_npz(pred_path)

    @abstractmethod
    def _run_script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        """The command that will run the script."""

    @abstractmethod
    def _fit_script_command(self, train_path: Path, model_path: Path) -> List[str]:
        """The command that will make the script fit."""

    @abstractmethod
    def _predict_script_command(
        self, model_path: Path, test_path: Path, pred_path: Path
    ) -> List[str]:
        """The command that will make the script predict."""
