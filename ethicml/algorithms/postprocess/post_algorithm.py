"""Abstract Base Class of all post-processing algorithms in the framework."""

from abc import abstractmethod

from ethicml.utility import DataTuple, Prediction, TestTuple

from ..algorithm_base import Algorithm

__all__ = ["PostAlgorithm"]


class PostAlgorithm(Algorithm):
    """Abstract Base Class for all algorithms that do post-processing."""

    @abstractmethod
    def run(
        self,
        train_predictions: Prediction,
        train: DataTuple,
        test_predictions: Prediction,
        test: TestTuple,
    ) -> Prediction:
        """Make predictions fair.

        Args:
            train_predictions: predictions on a training set
            train: the training set with the correct labels and sensitive attributes
            test_predictions: predictions on the test set
            test: the test set with the sensitive attributes

        Returns:
            post-processed predictions on the test set
        """
