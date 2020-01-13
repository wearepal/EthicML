"""Abstract Base Class of all post-processing algorithms in the framework."""

from abc import abstractmethod
import pandas as pd

from ethicml.utility.data_structures import DataTuple, TestTuple
from ..algorithm_base import Algorithm


class PostAlgorithm(Algorithm):
    """Abstract Base Class for all algorithms that do post-processing."""

    @abstractmethod
    def run(
        self,
        train_predictions: pd.DataFrame,
        train: DataTuple,
        test_predictions: pd.DataFrame,
        test: TestTuple,
    ) -> pd.DataFrame:
        """Make predictions fair.

        Args:
            train_predictions: predictions on a training set
            train: the training set with the correct labels and sensitive attributes
            test_predictions: predictions on the test set
            test: the test set with the sensitive attributes
        Return:
            post-processed predictions on the test set
        """
