"""Wrapper for SKLearn implementation of SVM."""
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
from ranzen import implements
from sklearn.svm import SVC, LinearSVC

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["SVM"]

from ethicml.utility import KernelType


@dataclass
class SVM(InAlgorithm):
    """A wrapper around the SciKitLearn Support Vector Classifier (SVC) model.

    Documentation for the underlying classifier can be found
    `here <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """

    C: float = field(default_factory=lambda: SVC().C)
    kernel: KernelType = field(default_factory=lambda: KernelType[SVC().kernel])

    def __post_init__(self) -> None:
        self._hyperparameters = {"C": self.C, "kernel": self.kernel}

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return f"SVM ({self.kernel})"

    @implements(InAlgorithm)
    def fit(self, train: DataTuple, seed: int) -> "SVM":
        self.clf = select_svm(self.C, self.kernel, seed)
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.clf.predict(test.x)))


def select_svm(C: float, kernel: KernelType, seed: int) -> Union[LinearSVC, SVC]:
    """Select the appropriate SVM model for the given parameters.

    :param C: The penalty parameter of the error term.
    :param kernel: The kernel to use.
    :param seed: The seed for the random number generator.
    """
    random_state = np.random.RandomState(seed=seed)
    if kernel is KernelType.linear:
        return LinearSVC(C=C, dual=False, tol=1e-12, random_state=random_state)
    return SVC(C=C, kernel=kernel.name, gamma="auto", random_state=random_state)
