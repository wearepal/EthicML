"""
Abstract Base Class of all algorithms in the framework
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, List
from abc import abstractmethod
import pandas as pd

from ethicml.algorithms.algorithm_base import Algorithm, AlgorithmAsync, run_blocking
from ..utils import (
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
    def run(self, train: DataTuple, test: TestTuple) -> (Tuple[pd.DataFrame, pd.DataFrame]):
        """Generate fair features with the given data

        Args:
            train: training data
            test: test data
        """

    def run_test(self, train: DataTuple, test: TestTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run with reduced training set so that it finishes quicker"""
        train_testing = get_subset(train)
        return self.run(train_testing, test)


class PreAlgorithmAsync(PreAlgorithm, AlgorithmAsync):
    """Pre-Algorithm that can be run blocking and asynchronously"""

    def run(self, train, test):
        return run_blocking(self.run_async(train, test))

    async def run_async(
        self, train: DataTuple, test: TestTuple
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate fair features with the given data asynchronously"""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_as_feather(train, test, tmp_path)
            train_path = tmp_path / "transform_train.feather"
            test_path = tmp_path / "transform_test.feather"
            cmd = self._script_command(train_paths, test_paths, train_path, test_path)
            await self._call_script(cmd)
            return load_feather(train_path), load_feather(test_path)

    @abstractmethod
    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_path: Path,
        new_test_path: Path,
    ) -> List[str]:
        """The command that will run the script"""
