"""Abstract Base Class of all algorithms in the framework."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar

from ethicml.models.algorithm_base import Algorithm
from ethicml.utility import DataTuple, SubgroupTuple

__all__ = ["PreAlgorithm"]

T = TypeVar("T", DataTuple, SubgroupTuple)
_PA = TypeVar("_PA", bound="PreAlgorithm")


class PreAlgorithm(Algorithm, ABC):
    """Abstract Base Class for all algorithms that do pre-processing."""

    @abstractmethod
    def fit(self: _PA, train: DataTuple, seed: int = 888) -> tuple[_PA, DataTuple]:
        """Fit transformer on the given data.

        :param train: Data tuple of the training data.
        :param seed: Random seed for model initialization.
        :returns: A tuple of Self and the test data.
        """

    @abstractmethod
    def transform(self, data: T) -> T:
        """Generate fair features with the given data.

        :param data: Data to transform.
        :returns: Transformed data.
        """

    @abstractmethod
    def run(self, train: DataTuple, test: T, seed: int = 888) -> tuple[DataTuple, T]:
        """Generate fair features with the given data.

        :param train: Data tuple of the training data.
        :param test: Data tuple of the test data.
        :param seed: Random seed for model initialization.
        :returns: A tuple of the transforme training data and the test data.
        """

    def run_test(self, train: DataTuple, test: T, seed: int = 888) -> tuple[DataTuple, T]:
        """Run with reduced training set so that it finishes quicker.

        :param train: Data tuple of the training data.
        :param test: Data tuple of the test data.
        :param seed: Random seed for model initialization.
        :returns: A tuple of the transforme training data and the test data.
        """
        train_testing = train.get_n_samples()
        return self.run(train_testing, test, seed)

    @property
    @abstractmethod
    def out_size(self) -> int:
        """Return the number of features to generate."""
