"""Wrapper for SKLearn implementation of SVM."""
from typing import Optional, Union

import numpy as np
import pandas as pd
from ranzen import implements
from sklearn.svm import SVC, LinearSVC

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["SVM"]


class SVM(InAlgorithm):
    """Support Vector Machine."""

    is_fairness_algo = False

    def __init__(self, C: Optional[float] = None, kernel: Optional[str] = None, seed: int = 888):
        kernel_name = f" ({kernel})" if kernel is not None else ""
        self.__name = "SVM" + kernel_name
        self.seed = seed
        self.C = SVC().C if C is None else C
        self.kernel = SVC().kernel if kernel is None else kernel

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return self.__name

    @implements(InAlgorithm)
    def fit(self, train: DataTuple) -> InAlgorithm:
        self.clf = select_svm(self.C, self.kernel, self.seed)
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.clf.predict(test.x)))


def select_svm(C: float, kernel: str, seed: int) -> Union[LinearSVC, SVC]:
    """Select the appropriate SVM model for the given parameters."""
    random_state = np.random.RandomState(seed=seed)
    if kernel == "linear":
        return LinearSVC(C=C, dual=False, tol=1e-12, random_state=random_state)
    return SVC(C=C, kernel=kernel, gamma="auto", random_state=random_state)
