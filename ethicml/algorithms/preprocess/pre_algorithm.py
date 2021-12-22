"""Abstract Base Class of all algorithms in the framework."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple, TypeVar

from ranzen import implements

from ethicml.algorithms.algorithm_base import Algorithm, AlgorithmAsync, run_blocking
from ethicml.utility import DataTuple, TestTuple

__all__ = ["PreAlgorithm", "PreAlgorithmAsync"]

T = TypeVar("T", DataTuple, TestTuple)


class PreAlgorithm(Algorithm):
    """Abstract Base Class for all algorithms that do pre-processing."""

    def __init__(self, name: str, seed: int, out_size: Optional[int]):
        """Base constructor for the Algorithm class.

        Args:
            name: name of the algorithm
            seed: seed for the random number generator
            out_size: number of features to generate
        """
        super().__init__(name=name, seed=seed)
        self._out_size = out_size

    @abstractmethod
    def fit(self, train: DataTuple) -> Tuple[PreAlgorithm, DataTuple]:
        """Generate fair features with the given data.

        Args:
            train: training data
            test: test data

        Returns:
            a tuple of the pre-processed training data and the test data
        """

    @abstractmethod
    def transform(self, data: T) -> T:
        """Generate fair features with the given data.

        Args:
            train: training data
            test: test data

        Returns:
            a tuple of the pre-processed training data and the test data
        """

    @abstractmethod
    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Generate fair features with the given data.

        Args:
            train: training data
            test: test data

        Returns:
            a tuple of the pre-processed training data and the test data
        """

    def run_test(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Run with reduced training set so that it finishes quicker."""
        train_testing = train.get_subset()
        return self.run(train_testing, test)

    @property
    def out_size(self) -> int:
        """The number of features to generate."""
        assert self._out_size is not None
        return self._out_size


class PreAlgorithmAsync(PreAlgorithm, AlgorithmAsync):
    """Pre-Algorithm that can be run blocking and asynchronously."""

    @implements(PreAlgorithm)
    def fit(self, train: DataTuple) -> Tuple[PreAlgorithm, DataTuple]:
        model, data = run_blocking(self.fit_async(train))
        return self, data

    @implements(PreAlgorithm)
    def transform(self, data: T) -> T:
        return run_blocking(self.transform_async(data))

    @implements(PreAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Generate fair features with the given data by running as a blocking function.

        Args:
            train: training data
            test: test data

        Returns:
            a tuple of the pre-processed training data and the test data
        """
        return run_blocking(self.run_async(train, test))

    async def fit_async(self, train: DataTuple) -> Tuple[PreAlgorithm, DataTuple]:
        """Generate fair features with the given data asynchronously.

        Args:
            train: training data
            test: test data

        Returns:
            a tuple of the pre-processed training data and the test data
        """
        self.model_path = self.model_dir / f"model_{self.name}.joblib"
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # ================================ write data to files ================================
            train_path, test_path = tmp_path / "train.npz", tmp_path / "test.npz"
            train.to_npz(train_path)

            # ========================== generate commandline arguments ===========================
            transformed_train_path = tmp_path / "transformed_train.npz"
            cmd = self._fit_script_command(train_path, transformed_train_path, self.model_path)

            # ============================= run the generated command =============================
            await self._call_script(cmd + ["--mode", "fit"])

            # ================================== load results =====================================
            transformed_train = DataTuple.from_npz(transformed_train_path)

        # prefix the name of the algorithm to the dataset name
        transformed_train = transformed_train.replace(
            name=None if train.name is None else f"{self.name}: {train.name}"
        )
        return self, transformed_train

    async def transform_async(self, data: T) -> T:
        """Generate fair features with the given data asynchronously.

        Args:
            train: training data
            test: test data

        Returns:
            a tuple of the pre-processed training data and the test data
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # ================================ write data to files ================================
            test_path = tmp_path / "test.npz"
            data.to_npz(test_path)

            # ========================== generate commandline arguments ===========================
            transformed_test_path = tmp_path / "transformed_test.npz"
            cmd = self._transform_script_command(
                model_path=self.model_path, test_path=test_path, new_test_path=transformed_test_path
            )

            # ============================= run the generated command =============================
            await self._call_script(cmd + ["--mode", "transform"])

            # ================================== load results =====================================
            transformed_test = TestTuple.from_npz(transformed_test_path)

        # prefix the name of the algorithm to the dataset name
        transformed_test = transformed_test.replace(
            name=None if data.name is None else f"{self.name}: {data.name}"
        )
        return transformed_test

    async def run_async(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Generate fair features with the given data asynchronously.

        Args:
            train: training data
            test: test data

        Returns:
            a tuple of the pre-processed training data and the test data
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
            cmd = self._run_script_command(
                train_path, test_path, transformed_train_path, transformed_test_path
            )

            # ============================= run the generated command =============================
            await self._call_script(cmd + ["--mode", "run"])

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

    @abstractmethod
    def _run_script_command(
        self, train_path: Path, test_path: Path, new_train_path: Path, new_test_path: Path
    ) -> List[str]:
        """The command that will run the script."""

    @abstractmethod
    def _fit_script_command(
        self, train_path: Path, new_train_path: Path, model_path: Path
    ) -> List[str]:
        """The command that will run the script."""

    @abstractmethod
    def _transform_script_command(
        self, model_path: Path, test_path: Path, new_test_path: Path
    ) -> List[str]:
        """The command that will run the script."""
