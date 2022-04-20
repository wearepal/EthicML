"""Abstract Base Class of all algorithms in the framework."""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple, TypeVar, Union
from typing_extensions import Literal, Protocol, TypeAlias, TypedDict, runtime_checkable

from ranzen import implements

from ethicml.algorithms.algorithm_base import Algorithm, SubprocessAlgorithmMixin
from ethicml.utility import DataTuple, TestTuple

__all__ = ["PreAlgoArgs", "PreAlgorithm", "PreAlgorithmAsync", "PreAlgorithmDC"]

T = TypeVar("T", DataTuple, TestTuple)
_PA = TypeVar("_PA", bound="PreAlgorithm")


@runtime_checkable
class PreAlgorithm(Algorithm, Protocol):
    """Abstract Base Class for all algorithms that do pre-processing."""

    _out_size: Optional[int]

    @abstractmethod
    def fit(self: _PA, train: DataTuple) -> Tuple[_PA, DataTuple]:
        """Fit transformer on the given data.

        :param train: training data
        :returns: a tuple of the pre-processed training data and the test data
        """

    @abstractmethod
    def transform(self, data: T) -> T:
        """Generate fair features with the given data.

        :param data:
        :returns: a tuple of the pre-processed training data and the test data
        """

    @abstractmethod
    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Generate fair features with the given data.

        :param train: training data
        :param test: test data
        :returns: a tuple of the pre-processed training data and the test data
        """

    def run_test(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Run with reduced training set so that it finishes quicker.

        :param train:
        :param test:
        """
        train_testing = train.get_n_samples()
        return self.run(train_testing, test)

    @property
    def out_size(self) -> int:
        """Return the number of features to generate."""
        assert self._out_size is not None
        return self._out_size


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class PreAlgorithmDC(PreAlgorithm):
    """PreAlgorithm dataclass base class."""

    _out_size = None  # this is not a dataclass field and so it must not have a type annotation
    seed: int = 888


class PreAlgoRunArgs(TypedDict):
    """Base arguments for the ``run`` function of async pre-process methods."""

    mode: Literal["run"]
    train: str
    test: str
    # paths to where the processed inputs should be stored
    new_train: str
    new_test: str


class PreAlgoFitArgs(TypedDict):
    """Base arguments for the ``fit`` function of async pre-process methods."""

    mode: Literal["fit"]
    train: str
    new_train: str
    model: str  # path to where the model weights are stored


class PreAlgoTformArgs(TypedDict):
    """Base arguments for the ``transform`` function of async pre-process methods."""

    mode: Literal["transform"]
    test: str
    new_test: str
    model: str


PreAlgoArgs: TypeAlias = Union[PreAlgoFitArgs, PreAlgoTformArgs, PreAlgoRunArgs]


class PreAlgorithmAsync(SubprocessAlgorithmMixin, PreAlgorithm, Protocol):
    """Pre-Algorithm that can be run blocking and asynchronously."""

    model_dir: Path

    @implements(PreAlgorithm)
    def fit(self, train: DataTuple) -> Tuple[PreAlgorithm, DataTuple]:
        """Generate fair features with the given data asynchronously.

        :param train: training data
        :returns: a tuple of the pre-processed training data and the test data
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # ================================ write data to files ================================
            train_path = tmp_path / "train.npz"
            train.to_npz(train_path)

            # ========================== generate commandline arguments ===========================
            transformed_train_path = tmp_path / "transformed_train.npz"
            args: PreAlgoFitArgs = {
                "mode": "fit",
                "model": str(self._model_path),
                "train": str(train_path),
                "new_train": str(transformed_train_path),
            }

            # ============================= run the generated command =============================
            self._call_script(self._script_command(args))

            # ================================== load results =====================================
            transformed_train = DataTuple.from_npz(transformed_train_path)

        # prefix the name of the algorithm to the dataset name
        transformed_train = transformed_train.replace(
            name=None if train.name is None else f"{self.name}: {train.name}"
        )
        return self, transformed_train

    @implements(PreAlgorithm)
    def transform(self, data: T) -> T:
        """Generate fair features with the given data asynchronously.

        :param data:
        :returns: a tuple of the pre-processed training data and the test data
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # ================================ write data to files ================================
            test_path = tmp_path / "test.npz"
            data.to_npz(test_path)

            # ========================== generate commandline arguments ===========================
            transformed_test_path = tmp_path / "transformed_test.npz"
            args: PreAlgoTformArgs = {
                "mode": "transform",
                "model": str(self._model_path),
                "test": str(test_path),
                "new_test": str(transformed_test_path),
            }

            # ============================= run the generated command =============================
            self._call_script(self._script_command(args))

            # ================================== load results =====================================
            transformed_test = TestTuple.from_npz(transformed_test_path)

        # prefix the name of the algorithm to the dataset name
        transformed_test = transformed_test.replace(
            name=None if data.name is None else f"{self.name}: {data.name}"
        )
        return transformed_test

    @implements(PreAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Generate fair features with the given data asynchronously.

        :param train: training data
        :param test: test data
        :returns: a tuple of the pre-processed training data and the test data
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # ================================ write data to files ================================
            train_path, test_path = tmp_path / "train.npz", tmp_path / "test.npz"
            train.to_npz(train_path)
            test.to_npz(test_path)

            # ========================== generate commandline arguments ===========================
            transformed_train_path = tmp_path / "transformed_train.npz"
            transformed_test_path = tmp_path / "transformed_test.npz"
            args: PreAlgoRunArgs = {
                "mode": "run",
                "train": str(train_path),
                "test": str(test_path),
                "new_train": str(transformed_train_path),
                "new_test": str(transformed_test_path),
            }

            # ============================= run the generated command =============================
            self._call_script(self._script_command(args))

            # ================================== load results =====================================
            transformed_train = DataTuple.from_npz(transformed_train_path)
            transformed_test = TestTuple.from_npz(transformed_test_path)

        # prefix the name of the algorithm to the dataset name
        transformed_train = transformed_train.replace(
            name=None if train.name is None else f"{self.name}: {train.name}"
        )
        transformed_test = transformed_test.replace(
            name=None if test.name is None else f"{self.name}: {test.name}"
        )
        return transformed_train, transformed_test

    @property
    def _model_path(self) -> Path:
        return self.model_dir / f"model_{self.name}.joblib"

    @abstractmethod
    def _script_command(self, pre_algo_args: PreAlgoArgs) -> List[str]:
        """Return the command that will run the script.

        :param pre_algo_args:
        """
