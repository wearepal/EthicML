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

        Args:
            train: training data

        Returns:
            self, but trained.
        """

    @abstractmethod
    def predict(self, test_predictions: Prediction, test: TestTuple) -> Prediction:
        """Run Algorithm on the given data.

        Args:
            test: data to evaluate on

        Returns:
            predictions
        """

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
        self.fit(train_predictions, train)
        return self.predict(test_predictions, test)


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class PostAlgorithmDC(PostAlgorithm):
    """PostAlgorithm dataclass base class."""

    is_fairness_algo = True
    seed: int = 888
