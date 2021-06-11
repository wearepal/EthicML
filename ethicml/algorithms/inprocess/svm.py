"""Wrapper for SKLearn implementation of SVM."""
from typing import Optional, Union

import pandas as pd
from kit import implements
from sklearn.svm import SVC, LinearSVC

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["SVM"]


class SVM(InAlgorithm):
    """Support Vector Machine."""

    def __init__(self, C: Optional[float] = None, kernel: Optional[str] = None):
        kernel_name = f" ({kernel})" if kernel is not None else ""
        super().__init__(name="SVM" + kernel_name, is_fairness_algo=False)
        self.C = SVC().C if C is None else C
        self.kernel = SVC().kernel if kernel is None else kernel

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: Union[DataTuple, TestTuple]) -> Prediction:
        clf = select_svm(self.C, self.kernel)
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict(test.x)))


def select_svm(C: float, kernel: str) -> SVC:
    """Select the appropriate SVM model for the given parameters."""
    if kernel == "linear":
        return LinearSVC(C=C, dual=False, tol=1e-12, random_state=888)
    return SVC(C=C, kernel=kernel, gamma="auto", random_state=888)
