"""Classes related to running algorithms in subprocesses."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Mapping, TypedDict, TypeVar, Union, final
from typing_extensions import TypeAlias

from ethicml.models.algorithm_base import SubprocessAlgorithmMixin
from ethicml.models.preprocess.pre_algorithm import PreAlgorithm
from ethicml.utility import DataTuple, SubgroupTuple

__all__ = ["PreAlgoArgs", "PreAlgorithmSubprocess"]

T = TypeVar("T", DataTuple, SubgroupTuple)


class PreAlgoRunArgs(TypedDict):
    """Base arguments for the ``run`` function of async pre-process methods."""

    mode: Literal["run"]
    train: str
    test: str
    # paths to where the processed inputs should be stored
    new_train: str
    new_test: str
    seed: int


class PreAlgoFitArgs(TypedDict):
    """Base arguments for the ``fit`` function of async pre-process methods."""

    mode: Literal["fit"]
    train: str
    new_train: str
    model: str  # path to where the model weights are stored
    seed: int


class PreAlgoTformArgs(TypedDict):
    """Base arguments for the ``transform`` function of async pre-process methods."""

    mode: Literal["transform"]
    test: str
    new_test: str
    model: str


PreAlgoArgs: TypeAlias = Union[PreAlgoFitArgs, PreAlgoTformArgs, PreAlgoRunArgs]
_P = TypeVar("_P", bound="PreAlgorithmSubprocess")


@dataclass
class PreAlgorithmSubprocess(SubprocessAlgorithmMixin, PreAlgorithm, ABC):
    """Pre-Algorithm that runs the method in a subprocess.

    This is the base class for all pre-processing algorithms that run in a subprocess. The
    advantage of this is that it allows for parallelization.
    """

    dir: Path = Path(".")

    @property
    @final
    def model_path(self) -> Path:
        """Path to where the model with be stored."""
        return self.dir.resolve(strict=True) / f"model_{self.name}.joblib"

    @final
    def fit(self: _P, train: DataTuple, seed: int = 888) -> tuple[_P, DataTuple]:
        """Fit transformer in a subprocess on the given data.

        :param train: Data tuple of the training data.
        :param seed: Random seed for model initialization.
        :returns: A tuple of Self and the test data.
        """
        self._in_size: int | None = train.x.shape[1]
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # ================================ write data to files ================================
            train_path = tmp_path / "train.npz"
            train.save_to_file(train_path)

            # ========================== generate commandline arguments ===========================
            transformed_train_path = tmp_path / "transformed_train.npz"
            args: PreAlgoFitArgs = {
                "mode": "fit",
                "model": str(self.model_path),
                "train": str(train_path),
                "new_train": str(transformed_train_path),
                "seed": seed,
            }

            # ============================= run the generated command =============================
            self.call_script(self._script_command(args))

            # ================================== load results =====================================
            transformed_train = DataTuple.from_file(transformed_train_path)

        # prefix the name of the algorithm to the dataset name
        if train.name is not None:
            transformed_train = transformed_train.rename(f"{self.name}: {train.name}")
        return self, transformed_train

    @final
    def transform(self, data: T) -> T:
        """Generate fair features in a subprocess with the given data.

        :param data: Data to transform.
        :returns: Transformed data.
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # ================================ write data to files ================================
            test_path = tmp_path / "test.npz"
            if isinstance(data, DataTuple):
                data.remove_y().save_to_file(test_path)
            else:
                data.save_to_file(test_path)

            # ========================== generate commandline arguments ===========================
            transformed_test_path = tmp_path / "transformed_test.npz"
            args: PreAlgoTformArgs = {
                "mode": "transform",
                "model": str(self.model_path),
                "test": str(test_path),
                "new_test": str(transformed_test_path),
            }

            # ============================= run the generated command =============================
            self.call_script(self._script_command(args))

            # ================================== load results =====================================
            transformed_test: T = _load_file(transformed_test_path, data)

        # prefix the name of the algorithm to the dataset name
        if data.name is not None:
            transformed_test = transformed_test.rename(f"{self.name}: {data.name}")
        return transformed_test

    @final
    def run(self, train: DataTuple, test: T, seed: int = 888) -> tuple[DataTuple, T]:
        """Generate fair features in a subprocess with the given data.

        :param train: Data tuple of the training data.
        :param test: Data tuple of the test data.
        :param seed: Random seed for model initialization.
        :returns: A tuple of the transforme training data and the test data.
        """
        self._in_size = train.x.shape[1]
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # ================================ write data to files ================================
            train_path, test_path = tmp_path / "train.npz", tmp_path / "test.npz"
            train.save_to_file(train_path)
            if isinstance(test, DataTuple):
                test.remove_y().save_to_file(test_path)
            else:
                test.save_to_file(test_path)

            # ========================== generate commandline arguments ===========================
            transformed_train_path = tmp_path / "transformed_train.npz"
            transformed_test_path = tmp_path / "transformed_test.npz"
            args: PreAlgoRunArgs = {
                "mode": "run",
                "train": str(train_path),
                "test": str(test_path),
                "new_train": str(transformed_train_path),
                "new_test": str(transformed_test_path),
                "seed": seed,
            }

            # ============================= run the generated command =============================
            self.call_script(self._script_command(args))

            # ================================== load results =====================================
            transformed_train = DataTuple.from_file(transformed_train_path)
            transformed_test = _load_file(transformed_test_path, test)

        # prefix the name of the algorithm to the dataset name
        if train.name is not None:
            transformed_train = transformed_train.rename(f"{self.name}: {train.name}")
        if test.name is not None:
            transformed_test = transformed_test.rename(f"{self.name}: {test.name}")
        return transformed_train, transformed_test

    @final
    def _script_command(self, pre_algo_args: PreAlgoArgs) -> list[str]:
        """Return the command that will run the script.

        The flag interface consists of two strings, both JSON strings: the general pre-algo flags
        and then the more specific flags for the algorithm.

        :param pre_algo_args: The Arguments that are shared among all pre-processing algorithms.
        :return: List of strings that can be passed to ``subprocess.run``.
        """
        interface = [
            json.dumps(pre_algo_args, separators=(',', ':')),
            json.dumps(self._get_flags(), separators=(',', ':')),
        ]
        return self._get_path_to_script() + interface

    @abstractmethod
    def _get_path_to_script(self) -> list[str]:
        """Return arguments that are passed to the python executable."""

    @abstractmethod
    def _get_flags(self) -> Mapping[str, Any]:
        """Return flags that are used to configure this algorithm."""


def _load_file(file_path: Path, original: T) -> T:
    loaded = SubgroupTuple.from_file(file_path)
    if isinstance(original, DataTuple):
        return DataTuple.from_df(x=loaded.x, s=loaded.s, y=original.y, name=loaded.name)
    else:
        return loaded
