"""Classes related to running algorithms in subprocesses."""
import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Mapping, TypeVar, Union
from typing_extensions import Literal, TypeAlias, TypedDict, final

from ethicml.algorithms.algorithm_base import SubprocessAlgorithmMixin
from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.utility.data_structures import DataTuple, Prediction, TestTuple

__all__ = ["InAlgoArgs", "InAlgorithmSubprocess"]


class InAlgoRunArgs(TypedDict):
    """Base arguments for the ``run`` function of subprocess in-process methods."""

    mode: Literal["run"]
    predictions: str  # path to where the predictions should be stored
    # paths to the files with the data
    train: str
    test: str
    seed: int


class InAlgoFitArgs(TypedDict):
    """Base arguments for the ``fit`` function of subprocess in-process methods."""

    mode: Literal["fit"]
    train: str
    model: str  # path to where the model weights are stored
    seed: int


class InAlgoPredArgs(TypedDict):
    """Base arguments for the ``predict`` function of subprocess in-process methods."""

    mode: Literal["predict"]
    predictions: str
    test: str
    model: str


InAlgoArgs: TypeAlias = Union[InAlgoFitArgs, InAlgoPredArgs, InAlgoRunArgs]


_IS = TypeVar("_IS", bound="InAlgorithmSubprocess")


@dataclass  # type: ignore[misc]  # mypy doesn't allow abstract dataclasses because mypy is stupid
class InAlgorithmSubprocess(SubprocessAlgorithmMixin, InAlgorithm):
    """In-Algorithm that uses a subprocess to run.

    :param dir: Directory to store the model.
    """

    dir: Path = Path(".")

    @property
    @final
    def model_path(self) -> Path:
        """Path to where the model with be stored."""
        return self.dir.resolve(strict=True) / f"model_{self.name}.joblib"

    @final
    def fit(self: _IS, train: DataTuple, seed: int = 888) -> _IS:
        """Fit Algorithm in a subprocess on the given data.

        :param train: Data tuple of the training data.
        :param seed: Random seed for model initialization.
        :returns: Self, but trained.
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = tmp_path / "train.npz"
            train.to_npz(train_path)
            args: InAlgoFitArgs = {
                "mode": "fit",
                "train": str(train_path),
                "model": str(self.model_path),
                "seed": seed,
            }
            self.call_script(self._script_command(args))
            return self

    @final
    def predict(self, test: TestTuple) -> Prediction:
        """Make predictions in a subprocess on the given data.

        :param test: Data to evaluate on.
        :returns: Predictions on the test data.
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_path = tmp_path / "test.npz"
            pred_path = tmp_path / "predictions.npz"
            test.to_npz(test_path)
            args: InAlgoPredArgs = {
                "mode": "predict",
                "test": str(test_path),
                "predictions": str(pred_path),
                "model": str(self.model_path),
            }
            self.call_script(self._script_command(args))
            return Prediction.from_npz(pred_path)

    @final
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        """Run Algorithm in a subprocess on the given data.

        :param train: Data tuple of the training data.
        :param test: Data to evaluate on.
        :param seed: Random seed for model initialization.
        :returns: Predictions on the test data.
        """
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
                "seed": seed,
            }
            self.call_script(self._script_command(args))
            return Prediction.from_npz(pred_path)

    @final
    def _script_command(self, in_algo_args: InAlgoArgs) -> List[str]:
        """Return the command that will run the script.

        The flag interface consists of two strings, both JSON strings: the general in-algo flags
        and then the more specific flags for the algorithm.

        :param in_algo_args: Arguments for the script.
        :returns: List of strings that will be passed to ``subprocess.run``.
        """
        interface = [
            json.dumps(in_algo_args, separators=(',', ':')),
            json.dumps(self._get_flags(), separators=(',', ':')),
        ]
        return self._get_path_to_script() + interface

    @abstractmethod
    def _get_path_to_script(self) -> List[str]:
        """Return arguments that are passed to the python executable."""

    @abstractmethod
    def _get_flags(self) -> Mapping[str, Any]:
        """Return flags that are used to configure this algorithm."""
