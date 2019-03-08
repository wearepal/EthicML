"""
Abstract Base Class of all algorithms in the framework
"""
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import numpy
import pandas as pd

from ethicml.algorithms.algorithm_base import Algorithm
from ethicml.algorithms.utils import DataTuple, get_subset, write_data_tuple


class InAlgorithm(Algorithm):
    """Abstract Base Class dor algorithms that run in the middle of the pipeline"""

    @abstractmethod
    def run(self, train: DataTuple, test: DataTuple, sub_process: bool = False) -> pd.DataFrame:
        """Run Algorithm

        Args:
            train:
            test:
            sub_process: indicate if the algorithm is to be run in it's own process
        """
        raise NotImplementedError("This method needs to be implemented in all models")

    def run_test(self, train: DataTuple, test: DataTuple) -> (
            pd.DataFrame):
        """

        Args:
            train:
            test:

        Returns:

        """
        train_testing = get_subset(train)
        return self.run(train_testing, test)

    def run_threaded(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        """ orchestrator for threaded """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_paths, test_paths = write_data_tuple(train, test, tmp_path)
            result = self.run_thread(train_paths, test_paths, tmp_path)
        return result

    def run_thread(self, train_paths, test_paths, tmp_path) -> pd.DataFrame:
        """ runs algorithm in its own thread """
        pred_path = tmp_path / "predictions.parquet"
        args = self._script_interface(train_paths, test_paths, pred_path)
        self._call_script(self.__module__, args)
        return self._load_output(pred_path)

    @staticmethod
    def _load_output(file_path: Path) -> pd.DataFrame:
        """Load a dataframe from a parquet file"""
        with file_path.open('rb') as file_obj:
            df = pd.read_parquet(file_obj)
        return df

    def save_predictions(self, predictions: Union[numpy.array, pd.DataFrame]):
        """Save the data to the file that was specified in the commandline arguments"""
        if not isinstance(predictions, pd.DataFrame):
            df = pd.DataFrame(predictions, columns=["pred"])
        else:
            df = predictions
        pred_path = Path(self.args[6])
        df.to_parquet(pred_path, compression=None)

    @staticmethod
    def _script_interface(train_paths, test_paths, *args):
        """
        Generate the commandline arguments that are expected by the scripts that follow the
        convention.

        The agreed upon order is:
        x (train), s (train), y (train), x (test), s (test), y (test), predictions.
        :param **kwargs:
        """
        list_to_return = [
            str(train_paths.x),
            str(train_paths.s),
            str(train_paths.y),
            str(test_paths.x),
            str(test_paths.s),
            str(test_paths.y)
        ]
        for arg in args:
            list_to_return.append(str(arg))
        return list_to_return
