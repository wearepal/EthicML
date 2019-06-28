"""
Abstract Base Class of all algorithms in the framework
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, List
from abc import abstractmethod

from ethicml.algorithms.algorithm_base import Algorithm, AlgorithmAsync, run_blocking
from ethicml.utility.data_structures import (
    get_subset,
    DataTuple,
    TestTuple,
    PathTuple,
    TestPathTuple,
    write_as_feather,
    load_feather,
)


class PreAlgorithm(Algorithm):
    """Abstract Base Class for all algorithms that do pre-processing"""

    @abstractmethod
    def run(self, train: DataTuple, test: TestTuple) -> (Tuple[DataTuple, TestTuple]):
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

    def run(self, train: DataTuple, test: TestTuple):
        return run_blocking(self.run_async(train, test))

    async def run_async(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        """Generate fair features with the given data asynchronously"""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_as_feather(train, test, tmp_path)
            train_x_path = tmp_path / "transform_train_x.feather"
            train_s_path = tmp_path / "transform_train_s.feather"
            train_y_path = tmp_path / "transform_train_y.feather"
            train_name_path = tmp_path / "transform_train_name.feather"
            test_x_path = tmp_path / "transform_test_x.feather"
            test_s_path = tmp_path / "transform_test_s.feather"
            test_name_path = tmp_path / "transform_test_name.feather"
            cmd = self._script_command(
                train_paths,
                test_paths,
                train_x_path,
                train_s_path,
                train_y_path,
                train_name_path,
                test_x_path,
                test_s_path,
                test_name_path,
            )
            await self._call_script(cmd)
            return (
                DataTuple(
                    x=load_feather(train_x_path),
                    s=load_feather(train_s_path),
                    y=load_feather(train_y_path),
                    name=str(load_feather(train_name_path)['0'][0]),
                ),
                TestTuple(
                    x=load_feather(test_x_path),
                    s=load_feather(test_s_path),
                    name=str(load_feather(test_name_path)['0'][0]),
                ),
            )

    @abstractmethod
    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_x_path: Path,
        new_train_s_path: Path,
        new_train_y_path: Path,
        new_train_name_path: Path,
        new_test_x_path: Path,
        new_test_s_path: Path,
        new_test_name_path: Path,
    ) -> List[str]:
        """The command that will run the script"""
