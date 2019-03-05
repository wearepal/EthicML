"""
Abstract Base Class of all algorithms in the framework
"""

from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple
import pandas as pd

from ethicml.common import ROOT_PATH
from ethicml.algorithms.algorithm_base import Algorithm
from ..utils import get_subset, DataTuple, write_data_tuple


class PreAlgorithm(Algorithm):
    """Abstract Base Class for all algorithms that do pre-processing"""
    @abstractmethod
    def run(self, train: DataTuple, test: DataTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate fair features

        Args:
            train:
            test:
        """
        raise NotImplementedError("Run needs to be implemented")

    def run_test(self, train: DataTuple, test: DataTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Args:
            train:
            test:

        Returns:

        """
        train_testing = get_subset(train)
        return self.run(train_testing, test)

    def run_threaded(self, train: DataTuple, test: DataTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ orchestrator for threaded """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_data_tuple(train, test, tmp_path)
            result = self.run_thread(train_paths, test_paths, tmp_path)
        return result

    def run_thread(self, train_paths, test_paths, tmp_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ runs algorithm in its own thread """
        train_path = tmp_path / "transform_train.parquet"
        test_path = tmp_path / "transform_test.parquet"
        args = self._script_interface(train_paths, test_paths, train_path, test_path)
        self._call_script(str(ROOT_PATH / "algorithms" / "preprocess" / self.filename), args)
        return self._load_output(train_path, test_path)

    @staticmethod
    def _load_output(filepath_tr: Path, filepath_te: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load a dataframe from a parquet file"""
        with filepath_tr.open('rb') as file_obj:
            df_train = pd.read_parquet(file_obj)
        with filepath_te.open('rb') as file_obj:
            df_test = pd.read_parquet(file_obj)
        return df_train, df_test

    def save_transformations(self, transforms: Tuple[pd.DataFrame, pd.DataFrame]):
        """Save the data to the file that was specified in the commandline arguments"""
        transform_train, transform_test = transforms
        if not isinstance(transform_train, pd.DataFrame):
            raise AssertionError()
        elif not isinstance(transform_test, pd.DataFrame):
            raise AssertionError()
        else:
            transform_train_path = Path(self.args[6])
            transform_train.columns = transform_train.columns.astype(str)
            transform_train.to_parquet(transform_train_path, compression=None)
            transform_test_path = Path(self.args[7])
            transform_test.columns = transform_test.columns.astype(str)
            transform_test.to_parquet(transform_test_path, compression=None)
