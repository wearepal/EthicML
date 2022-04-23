"""Abstract Base Class of all algorithms in the framework."""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Dict, List, TypeVar, Union
from typing_extensions import Literal, Protocol, TypeAlias, TypedDict, runtime_checkable

from ranzen import implements

from ethicml.algorithms.algorithm_base import Algorithm, SubprocessAlgorithmMixin
from ethicml.utility import DataTuple, FairnessType, KernelType, Prediction, TestTuple

__all__ = ["InAlgoArgs", "InAlgorithm", "InAlgorithmAsync", "InAlgorithmDC"]

_I = TypeVar("_I", bound="InAlgorithm")


@runtime_checkable
class InAlgorithm(Algorithm, Protocol):
    """Abstract Base Class for algorithms that run in the middle of the pipeline."""

    is_fairness_algo: ClassVar[bool]
    _hyperparameters: Dict[str, Union[str, int, float, KernelType, FairnessType]] = {}

    @abstractmethod
    def fit(self: _I, train: DataTuple) -> _I:
        """Fit Algorithm on the given data.

        :param train: Data tuple of the training data.
        :returns: Self, but trained.
        """

    @property
    def hyperparameters(
        self,
    ) -> Dict[str, Union[str, Any]]:
        """Return list of hyperparameters."""
        return self._hyperparameters

    @abstractmethod
    def predict(self, test: TestTuple) -> Prediction:
        """Make predictions on the given data.

        :param test: Data to evaluate on.
        :returns: Predictions on the test data.
        """

    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data.

        :param train: Data tuple of the training data.
        :param test: Data to evaluate on.
        :returns: Predictions on the test data.
        """
        self.fit(train)
        return self.predict(test)

    def run_test(self, train: DataTuple, test: TestTuple) -> Prediction:
        """Run with reduced training set so that it finishes quicker.

        :param train: Data tuple of the training data.
        :param test: Data to evaluate on.
        """
        train_testing = train.get_n_samples()
        return self.run(train_testing, test)


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class InAlgorithmDC(InAlgorithm):
    """InAlgorithm dataclass base class."""

    is_fairness_algo: ClassVar[bool] = True
    seed: int = 888


class InAlgoRunArgs(TypedDict):
    """Base arguments for the ``run`` function of async in-process methods."""

    mode: Literal["run"]
    predictions: str  # path to where the predictions should be stored
    # paths to the files with the data
    train: str
    test: str


class InAlgoFitArgs(TypedDict):
    """Base arguments for the ``fit`` function of async in-process methods."""

    mode: Literal["fit"]
    train: str
    model: str  # path to where the model weights are stored


class InAlgoPredArgs(TypedDict):
    """Base arguments for the ``predict`` function of async in-process methods."""

    mode: Literal["predict"]
    predictions: str
    test: str
    model: str


InAlgoArgs: TypeAlias = Union[InAlgoFitArgs, InAlgoPredArgs, InAlgoRunArgs]


_IA = TypeVar("_IA", bound="InAlgorithmAsync")


class InAlgorithmAsync(SubprocessAlgorithmMixin, InAlgorithm, Protocol):
    """In-Algorithm that uses a subprocess to run."""

    model_dir: Path

    @implements(InAlgorithm)
    def fit(self: _IA, train: DataTuple) -> _IA:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = tmp_path / "train.npz"
            train.to_npz(train_path)
            args: InAlgoFitArgs = {
                "mode": "fit",
                "train": str(train_path),
                "model": str(self._model_path),
            }
            self._call_script(self._script_command(args))
            return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_path = tmp_path / "test.npz"
            pred_path = tmp_path / "predictions.npz"
            test.to_npz(test_path)
            args: InAlgoPredArgs = {
                "mode": "predict",
                "test": str(test_path),
                "predictions": str(pred_path),
                "model": str(self._model_path),
            }
            self._call_script(self._script_command(args))
            return Prediction.from_npz(pred_path)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = tmp_path / "train.npz"
            test_path = tmp_path / "test.npz"
            pred_path = tmp_path / "predictions.npz"
            train.to_npz(train_path)
            test.to_npz(test_path)
            args: InAlgoRunArgs = {
                "mode": "run",
                "train": str(train_path),
                "test": str(test_path),
                "predictions": str(pred_path),
            }
            self._call_script(self._script_command(args))
            return Prediction.from_npz(pred_path)

    @property
    def _model_path(self) -> Path:
        return self.model_dir / f"model_{self.name}.joblib"

    @abstractmethod
    def _script_command(self, in_algo_args: InAlgoArgs) -> List[str]:
        """Return the command that will run the script.

        :param in_algo_args: Arguments for the script.
        :returns: List of strings that will be passed to ``subprocess.run``.
        """
