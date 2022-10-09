"""Wrapper for SKLearn implementation of SVM."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from ranzen import implements
from sklearn.svm import SVC, LinearSVC

from ethicml.models.inprocess.in_algorithm import InAlgorithmDC
from ethicml.utility import DataTuple, Prediction, TestTuple

__all__ = ["SVM"]

from ethicml.utility import KernelType


@dataclass
class SVM(InAlgorithmDC):
    """A wrapper around the SciKitLearn Support Vector Classifier (SVC) model.

    Documentation for the underlying classifier can be found
    `here <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.

    :param C: The penalty parameter of the error term.
    :param kernel: The kernel to use.
    """

    is_fairness_algo: ClassVar[bool] = False
    C: float = field(default_factory=lambda: SVC().C)
    kernel: KernelType = field(default_factory=lambda: KernelType[SVC().kernel])

    @property
    @implements(InAlgorithmDC)
    def name(self) -> str:
        return f"SVM ({self.kernel})"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple, seed: int = 888) -> SVM:
        self.clf = select_svm(self.C, self.kernel, seed)
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.clf.predict(test.x)))

    @implements(InAlgorithmDC)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        clf = select_svm(self.C, self.kernel, seed)
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict(test.x)))


def select_svm(C: float, kernel: KernelType, seed: int) -> LinearSVC | SVC:
    """Select the appropriate SVM model for the given parameters.

    :param C: The penalty parameter of the error term.
    :param kernel: The kernel to use.
    :param seed: The seed for the random number generator.
    :returns: The instantiated SVM.
    """
    random_state = np.random.RandomState(seed=seed)
    if kernel is KernelType.linear:
        return LinearSVC(C=C, dual=False, tol=1e-12, random_state=random_state)
    return SVC(C=C, kernel=kernel.name, gamma="auto", random_state=random_state)
