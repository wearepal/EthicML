"""
Base class for algorithms that run in their own thread
"""
from abc import abstractmethod
from typing import Optional, List
from pathlib import Path

import numpy as np
import pandas as pd

from ethicml.algorithms.algorithm_base import ThreadedAlgorithm
from ethicml.algorithms.utils import PathTuple


class ThreadedInAlgorithm(ThreadedAlgorithm):
    """Abstract Base Class for algorithms that run in the middle of the pipeline"""
    @abstractmethod
    def run(self, train_paths: PathTuple, test_paths: PathTuple, tmp_path: Path) -> pd.DataFrame:
        """Run Algorithm on the data that can be loaded from the given file paths

        Args:
            train_paths: Tuple of paths for the training data
            test_paths: Tuple of paths for the test data
            tmp_path: a path to a directory where the algorithm can write something
        """


class SimpleTIA(ThreadedInAlgorithm):
    """
    Simple threaded in-algorithm that just calls a single script

    The path to the script must be given in the __init__ method. The commandline arguments for the
    script are defined by subclasses of this class by overriding the script_interface method.
    """
    def __init__(self, name: str, script_path: str, executable: Optional[str] = None):
        """Constructor for the simple TIA

        Args:
            name: name of the algorithm
            script_path: path to the script that runs the algorithm
            executable: executable to run the script
        """
        super().__init__(name, executable=executable)
        self.script_path = script_path

    def run(self, train_paths, test_paths, tmp_path):
        # path where the predictions are supposed to be stored
        pred_path = tmp_path / "predictions.npy"
        # get the strings that are passed to the script as commandline arguments
        args = self.script_interface(train_paths, test_paths, pred_path)
        # call the script (this is blocking)
        self._call_script(self.script_path, args)
        # load the results
        predictions = np.load(pred_path)
        return pd.DataFrame(predictions, columns=["preds"])

    @abstractmethod
    def script_interface(self, train_paths: PathTuple, test_paths: PathTuple, pred_path: Path
                         ) -> List[str]:
        """Generate the commandline arguments that are expected by the script"""
