"""Abstract Base Class of all algorithms in the framework."""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, TypeVar
from typing_extensions import Protocol, runtime_checkable

from ranzen import implements

from ethicml.algorithms.algorithm_base import Algorithm, SubprocessAlgorithmMixin
from ethicml.utility import DataTuple, Prediction, TestTuple

__all__ = ["InAlgorithm", "InAlgorithmAsync", "InAlgorithmDC"]

_I = TypeVar("_I", bound="InAlgorithm")


@runtime_checkable
class InAlgorithm(Algorithm, Protocol):
    """Abstract Base Class for algorithms that run in the middle of the pipeline."""

    is_fairness_algo: bool

    @abstractmethod
    def fit(self: _I, train: DataTuple) -> _I:
        """Fit Algorithm on the given data.

        Args:
            train: training data

        Returns:
            self, but trained.
        """

    @abstractmethod
    def predict(self, test: TestTuple) -> Prediction:
        """Make predictions on the given data.

        Args:
            test: data to evaluate on

        Returns:
            predictions
        """

    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data.

        Args:
            train: training data
            test: test data

        Returns:
            predictions
        """
        self.fit(train)
        return self.predict(test)

    def run_test(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run with reduced training set so that it finishes quicker."""
        train_testing = train.get_subset()
        return self.run(train_testing, test)


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class InAlgorithmDC(InAlgorithm):
    """InAlgorithm dataclass base class."""

    is_fairness_algo = True
    seed: int = 888


_IA = TypeVar("_IA", bound="InAlgorithmAsync")


class InAlgorithmAsync(SubprocessAlgorithmMixin, InAlgorithm, Protocol):
    """In-Algorithm that uses a subprocess to run."""

    model_dir: Path

    @implements(InAlgorithm)
    def fit(self: _IA, train: DataTuple) -> _IA:
        """Fit algorithm on the given data asynchronously.

        Args:
            train: training data
            test: test data

        Returns:
            predictions
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = tmp_path / "train.npz"
            train.to_npz(train_path)
            cmd = self._fit_script_command(train_path, self._model_path)
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
            cmd = self._predict_script_command(self._model_path, test_path, pred_path)
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

    @property
    def _model_path(self) -> Path:
        return self.model_dir / f"model_{self.name}.joblib"

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
