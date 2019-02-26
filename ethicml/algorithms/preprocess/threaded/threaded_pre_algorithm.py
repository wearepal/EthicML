"""
Base class for algorithms that run in their own thread
"""
from abc import abstractmethod
from pathlib import Path
from typing import Tuple, List

import pandas as pd

from ethicml.algorithms.algorithm_base import ThreadedAlgorithm
from ethicml.algorithms.utils import PathTuple


class ThreadedPreAlgorithm(ThreadedAlgorithm):
    """This class only defines an interface for the threaded algorithms that do pre-processing.

    Most use cases will not subclass this but will subclass BasicTPA instead. (It might be
    possible to merge these two classes.)
    """
    @abstractmethod
    def run(self, train_paths: PathTuple, test_paths: PathTuple, tmp_path: Path) -> (
            Tuple[pd.DataFrame, pd.DataFrame]):
        """Generate fair features

        Args:
            train_paths: Tuple of paths for the training data
            test_paths: Tuple of paths for the test data
            tmp_path: a path to a directory where the algorithm can write something

        Returns:
            The transformed features for the training set and for the test set
        """


class BasicTPA(ThreadedPreAlgorithm):
    """Basic threaded pre-algorithm (TPA) that just calls a single script.

    The path to the script must be given in the __init__ method. The commandline arguments for the
    script are defined by subclasses of this class by overriding the script_interface method.
    """
    def __init__(self, name: str, script_path: str, **kwargs):
        """Constructor for the basic TPA

        Args:
            name: name of the algorithm
            script_path: path to the script that runs the algorithm
            kwargs: arguments that are passed on to ThreadedPreAlgorithm
        """
        super().__init__(name, **kwargs)
        self.script_path = script_path

    def run(self, train_paths, test_paths, tmp_path):
        # path where the pre-processed input for training and for testing should be stored
        for_train_path = tmp_path / "for_train.parquet"
        for_test_path = tmp_path / "for_test.parquet"
        # get the strings that are passed to the script as commandline arguments
        args = self._script_interface(train_paths, test_paths, for_train_path, for_test_path)
        # call the script (this is blocking)
        self._call_script(self.script_path, args)
        # load the results
        return self._load_output(for_train_path), self._load_output(for_test_path)

    @abstractmethod
    def _script_interface(self, train_paths: PathTuple, test_paths: PathTuple, for_train_path: Path,
                          for_test_path: Path) -> List[str]:
        """Generate the commandline arguments that are expected by the script"""
