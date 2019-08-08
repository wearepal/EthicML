"""
Abstract Base Class of all algorithms in the framework
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, List
from abc import abstractmethod
from dataclasses import replace

from ethicml.algorithms.algorithm_base import Algorithm, AlgorithmAsync, run_blocking
from ethicml.utility.data_structures import (
    get_subset,
    DataTuple,
    TestTuple,
    PathTuple,
    TestPathTuple,
    write_as_feather,
)


class PreAlgorithm(Algorithm):
    """Abstract Base Class for all algorithms that do pre-processing"""

    @abstractmethod
    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Generate fair features with the given data

        Args:
            train: training data
            test: test data
        """

    def run_test(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Run with reduced training set so that it finishes quicker"""
        train_testing = get_subset(train)
        return self.run(train_testing, test)


class PreAlgorithmAsync(PreAlgorithm, AlgorithmAsync):
    """Pre-Algorithm that can be run blocking and asynchronously"""

    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        return run_blocking(self.run_async(train, test))

    async def run_async(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Generate fair features with the given data asynchronously"""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # write data to files
            train_paths, test_paths = write_as_feather(train, test, tmp_path)

            # ========================== generate commandline arguments ===========================
            transformed_train_paths = PathTuple(
                x=tmp_path / "transform_train_x.feather",
                s=tmp_path / "transform_train_s.feather",
                y=tmp_path / "transform_train_y.feather",
                name=train.name if train.name is not None else "",
            )
            transformed_test_paths = TestPathTuple(
                x=tmp_path / "transform_test_x.feather",
                s=tmp_path / "transform_test_s.feather",
                name=test.name if test.name is not None else "",
            )
            cmd = self._script_command(
                train_paths, test_paths, transformed_train_paths, transformed_test_paths
            )

            # ============================= run the generated command =============================
            await self._call_script(cmd)

            # ================================== load results =====================================
            transformed_train = transformed_train_paths.load_from_feather()
            transformed_test = transformed_test_paths.load_from_feather()

        # prefix the name of the algorithm to the dataset name
        transformed_train = replace(
            transformed_train, name=None if train.name is None else f"{self.name}: {train.name}"
        )
        transformed_test = replace(
            transformed_test, name=None if test.name is None else f"{self.name}: {test.name}"
        )
        return transformed_train, transformed_test

    @abstractmethod
    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_paths: PathTuple,
        new_test_paths: TestPathTuple,
    ) -> List[str]:
        """The command that will run the script"""
