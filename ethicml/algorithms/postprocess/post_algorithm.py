"""Abstract Base Class of all post-processing algorithms in the framework."""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar
from typing_extensions import Protocol, runtime_checkable

from ethicml.utility import DataTuple, Prediction, TestTuple

from ..algorithm_base import Algorithm

__all__ = ["PostAlgorithm", "PostAlgorithmDC"]

_PA = TypeVar("_PA", bound="PostAlgorithm")


@runtime_checkable
class PostAlgorithm(Algorithm, Protocol):
    """Abstract Base Class for all algorithms that do post-processing."""

    @abstractmethod
    def fit(self: _PA, train_predictions: Prediction, train: DataTuple) -> _PA:
        """Run Algorithm on the given data.

        :param train_predictions:
        :param train: training data
        :returns: self, but trained.
        """

    @abstractmethod
    def predict(self, test_predictions: Prediction, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data.

        :param test_predictions:
        :param test: data to evaluate on
        :returns: predictions
        """

    def run(
        self,
        train_predictions: Prediction,
        train: DataTuple,
        test_predictions: Prediction,
        test: TestTuple,
    ) -> Prediction:
        """Make predictions fair.

        :param train_predictions: predictions on a training set
        :param train: the training set with the correct labels and sensitive attributes
        :param test_predictions: predictions on the test set
        :param test: the test set with the sensitive attributes
        :returns: post-processed predictions on the test set
        """
        self.fit(train_predictions, train)
        return self.predict(test_predictions, test)


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class PostAlgorithmDC(PostAlgorithm):
    """PostAlgorithm dataclass base class."""

    seed: int = 888
