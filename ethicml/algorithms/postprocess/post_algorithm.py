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

        :param train_predictions: Predictions on a training set.
        :param train: The training set with the correct labels and sensitive attributes.
        :returns: Self.
        """

    @abstractmethod
    def predict(self, test_predictions: Prediction, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data.

        :param test_predictions: Predictions on a test set.
        :param test: The test set with the sensitive attributes.
        :returns: Post-processed predictions on the test set.
        """

    def run(
        self,
        train_predictions: Prediction,
        train: DataTuple,
        test_predictions: Prediction,
        test: TestTuple,
    ) -> Prediction:
        """Make predictions fair.

        :param train_predictions: Predictions on a training set.
        :param train: The training set with the correct labels and sensitive attributes.
        :param test_predictions: Predictions on the test set.
        :param test: The test set with the sensitive attributes.
        :returns: Post-processed predictions on the test set.
        """
        self.fit(train_predictions, train)
        return self.predict(test_predictions, test)


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class PostAlgorithmDC(PostAlgorithm):
    """PostAlgorithm dataclass base class."""

    seed: int = 888
