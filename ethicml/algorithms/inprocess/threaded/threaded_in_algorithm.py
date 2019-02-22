"""
Base class for algorithms that run in their own thread
"""
from abc import abstractmethod
from pathlib import Path

import pandas as pd

from ethicml.algorithms.algorithm_base import ThreadedAlgorithm
from ethicml.algorithms.utils import PathTuple


class ThreadedInAlgorithm(ThreadedAlgorithm):
    """Abstract Base Class for algorithms that run in the middle of the pipeline"""

    @abstractmethod
    def run(self, train_paths: PathTuple, test_paths: PathTuple, tmp_path: Path) -> pd.DataFrame:
        """Run Algorithm on the data that can be loaded from the given file paths

        Args:
            train_paths: Dictionary of paths for the training data; the keys are: 'x', 's' and 'y'
            test_paths: Dictionary of paths for the test data; the keys are: 'x', 's' and 'y'
            tmp_path: a path to a directory where the algorithm can write something
        """
