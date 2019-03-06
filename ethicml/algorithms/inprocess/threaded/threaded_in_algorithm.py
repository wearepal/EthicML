"""
Base class for algorithms that run in their own thread
"""
from abc import abstractmethod
from typing import List
from pathlib import Path

import pandas as pd

from ethicml.algorithms.algorithm_base import ThreadedAlgorithm
from ethicml.algorithms.utils import PathTuple


class ThreadedInAlgorithm(ThreadedAlgorithm):
    """This class only defines an interface for the threaded algorithms that are in process.

    Most use cases will not subclass this but will subclass BasicTIA instead. (It might be
    possible to merge these two classes.)
    """
    @abstractmethod
    def run(self, train_paths: PathTuple, test_paths: PathTuple, tmp_path: Path) -> pd.DataFrame:
        """Run Algorithm on the data that can be loaded from the given file paths

        Args:
            train_paths: Tuple of paths for the training data
            test_paths: Tuple of paths for the test data
            tmp_path: a path to a directory where the algorithm can write something
        """


class BasicTIA(ThreadedInAlgorithm):
    """Basic threaded in-algorithm (TIA) that just calls a single script.

    The path to the script must be given in the __init__ method. The commandline arguments for the
    script are defined by subclasses of this class by overriding the script_interface method.
    """
    def __init__(self, name: str, script_path: str, **kwargs):
        """Constructor for the basic TIA

        Args:
            name: name of the algorithm
            script_path: path to the script that runs the algorithm
            kwargs: arguments that are passed on to ThreadedInAlgorithm
        """
        super().__init__(name, **kwargs)
        self.script_path = script_path

    def run(self, train_paths, test_paths, tmp_path):
        # path where the predictions are supposed to be stored
        pred_path = tmp_path / "predictions.parquet"
        # get the strings that are passed to the script as commandline arguments
        args = self._script_interface(train_paths, test_paths, pred_path)
        # call the script (this is blocking)
        self._call_script(self.script_path, args)
        # load the results
        return self._load_output(pred_path)


    @abstractmethod
    def _script_interface(self, train_paths: PathTuple, test_paths: PathTuple, pred_path: Path
                          ) -> List[str]:
        """Generate the commandline arguments that are expected by the script"""
