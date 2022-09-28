"""Classes related to running algorithms in subprocesses."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
import json
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Literal, Mapping, TypedDict, TypeVar, Union, final
from typing_extensions import TypeAlias
import uuid

from ethicml.models.algorithm_base import SubprocessAlgorithmMixin
from ethicml.models.inprocess.in_algorithm import InAlgorithm
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


@dataclass
class InAlgorithmSubprocess(SubprocessAlgorithmMixin, InAlgorithm, ABC):
    """In-Algorithm that uses a subprocess to run.

    :param dir: Directory to store the model.
    """

    dir: Path = field(default_factory=lambda: Path(gettempdir()))

    @cached_property  # needs to be cached because of the uuid4() call
    def model_path(self) -> Path:
        """Path to where the model with be stored."""
        name = self.name.replace(" ", "_")
        return self.dir.resolve(strict=True) / f"model_{name}_{uuid.uuid4()}.joblib"

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
            train.save_to_file(train_path)
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
            test.save_to_file(test_path)
            args: InAlgoPredArgs = {
                "mode": "predict",
                "test": str(test_path),
                "predictions": str(pred_path),
                "model": str(self.model_path),
            }
            self.call_script(self._script_command(args))
            return Prediction.from_file(pred_path)

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
            train.save_to_file(train_path)
            test.save_to_file(test_path)
            args: InAlgoRunArgs = {
                "mode": "run",
                "train": str(train_path),
                "test": str(test_path),
                "predictions": str(pred_path),
                "seed": seed,
            }
            self.call_script(self._script_command(args))
            return Prediction.from_file(pred_path)

    @final
    def _script_command(self, in_algo_args: InAlgoArgs) -> list[str]:
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
    def _get_path_to_script(self) -> list[str]:
        """Return arguments that are passed to the python executable."""

    @abstractmethod
    def _get_flags(self) -> Mapping[str, Any]:
        """Return flags that are used to configure this algorithm."""
