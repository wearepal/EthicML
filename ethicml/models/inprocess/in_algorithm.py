"""Abstract Base Class of all algorithms in the framework."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import TYPE_CHECKING, ClassVar, final
from typing_extensions import Self

from ethicml.models.algorithm_base import Algorithm
from ethicml.utility import DataTuple, HyperParamType, Prediction, TestTuple

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = ["InAlgorithm", "InAlgorithmDC", "InAlgorithmNoParams"]


class InAlgorithm(Algorithm, ABC):
    """Abstract Base Class for algorithms that run in the middle of the pipeline."""

    is_fairness_algo: ClassVar[bool] = True  # should be overwritten by subclasses

    @abstractmethod
    def fit(self, train: DataTuple, seed: int = 888) -> Self:
        """Fit Algorithm on the given data.

        :param train: Data tuple of the training data.
        :param seed: Random seed for model initialization.
        :returns: Self, but trained.
        """

    @abstractmethod
    def predict(self, test: TestTuple) -> Prediction:
        """Make predictions on the given data.

        :param test: Data to evaluate on.
        :returns: Predictions on the test data.
        """

    @abstractmethod
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        """Run Algorithm on the given data.

        :param train: Data tuple of the training data.
        :param test: Data to evaluate on.
        :param seed: Random seed for model initialization.
        :returns: Predictions on the test data.
        """

    @final
    def run_test(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        """Run with reduced training set so that it finishes quicker.

        :param train: Data tuple of the training data.
        :param test: Data to evaluate on.
        :param seed: Random seed for model initialization.
        :returns: Predictions on the test data.
        """
        train_testing = train.get_n_samples()
        return self.run(train_testing, test, seed)

    @property
    @abstractmethod
    def hyperparameters(self) -> HyperParamType:
        """Return list of hyperparameters."""


# ======== base classes with different default implementations of ``hyperparameters`` =========
class InAlgorithmNoParams(InAlgorithm, ABC):
    """Base class for algorithms without parameters."""

    @property
    @final
    def hyperparameters(self) -> HyperParamType:
        """Return list of hyperparameters."""
        return {}


class InAlgorithmDC(InAlgorithm, ABC):
    """Base class for algorithms that are dataclasses."""

    @property
    @final
    def hyperparameters(self: DataclassInstance) -> HyperParamType:
        """Return list of hyperparameters."""
        return asdict(self)
