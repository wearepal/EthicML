"""Abstract Base Class of all post-processing algorithms in the framework."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar

from ethicml.utility import DataTuple, Prediction, TestTuple

from ..algorithm_base import Algorithm

__all__ = ["PostAlgorithm"]

_PA = TypeVar("_PA", bound="PostAlgorithm")


class PostAlgorithm(Algorithm, ABC):
    """Abstract Base Class for all algorithms that do post-processing."""

    @abstractmethod
    def fit(self: _PA, train_predictions: Prediction, train: DataTuple) -> _PA:
        """Run Algorithm on the given data.

        :param train_predictions: Predictions on a training set.
        :param train: The training set with the correct labels and sensitive attributes.
        :returns: Self.
        """

    @abstractmethod
    def predict(self, test_predictions: Prediction, test: TestTuple, seed: int = 888) -> Prediction:
        """Run Algorithm on the given data.

        :param test_predictions: Predictions on a test set.
        :param test: The test set with the sensitive attributes.
        :param seed: The random seed.
        :returns: Post-processed predictions on the test set.
        """

    @abstractmethod
    def run(
        self,
        train_predictions: Prediction,
        train: DataTuple,
        test_predictions: Prediction,
        test: TestTuple,
        seed: int = 888,
    ) -> Prediction:
        """Make predictions fair.

        :param train_predictions: Predictions on a training set.
        :param train: The training set with the correct labels and sensitive attributes.
        :param test_predictions: Predictions on the test set.
        :param test: The test set with the sensitive attributes.
        :param seed: The random seed.
        :returns: Post-processed predictions on the test set.
        """
