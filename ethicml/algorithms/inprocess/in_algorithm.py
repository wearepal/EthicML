"""
Abstract Base Class of all algorithms in the framework
"""
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd

from ethicml.algorithms.algorithm_base import Algorithm
from ethicml.algorithms.utils import DataTuple, get_subset, PathTuple


class InAlgorithm(Algorithm):
    """Abstract Base Class dor algorithms that run in the middle of the pipeline"""

    def run(self, train: DataTuple, test: DataTuple, sub_process: bool = False) -> pd.DataFrame:
        """Run Algorithm either in process or out of process

        Args:
            train: training data
            test: test data
            sub_process: indicate if the algorithm is to be run in it's own process
        """
        return run_threaded(self, train, test) if sub_process else self._run(train, test)

    @abstractmethod
    def _run(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        """Run Algorithm on the given data"""
        raise NotImplementedError("`_run` needs to be implemented")

    def run_test(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        """Run with reduced training set so that it finishes quicker"""
        train_testing = get_subset(train)
        return self.run(train_testing, test)

    def run_thread(self, train_paths: PathTuple, test_paths: PathTuple, tmp_path: Path) -> Path:
        """ runs algorithm in its own thread """
        pred_path = tmp_path / "predictions.feather"
        cmd = self._script_command(train_paths, test_paths, pred_path)
        self._call_script(cmd)
        return pred_path

    def _script_command(self, train_paths: PathTuple, test_paths: PathTuple, pred_path: Path) -> (
            List[str]):
        """The command that will run the script"""
        raise NotImplementedError("`_script_command` is  not implemented")

    @staticmethod
    def _conventional_interface(train_paths: PathTuple, test_paths: PathTuple, *args) -> List[str]:
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


def run_threaded(model: InAlgorithm, train: DataTuple, test: DataTuple) -> pd.DataFrame:
    """Orchestrator for threaded

    This function is not a member function of InAlgorithm so that this function can never be
    overwritten.
    """
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        train_paths, test_paths = model.write_data(train, test, tmp_path)
        pred_path = model.run_thread(train_paths, test_paths, tmp_path)
        return model.load_output(pred_path)
