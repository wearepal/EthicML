"""Wrapper for SKLearn implementation of SVM."""
from typing import ClassVar, Optional, Union

import numpy as np
import pandas as pd
from ranzen import implements
from sklearn.svm import SVC, LinearSVC

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["SVM"]

from ...utility import KernelType


class SVM(InAlgorithm):
    """A wraper around the SciKitLearn Support Vector Classifier (SVC) model.

    Documentation for the underlying classifier can be found `here <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """

    is_fairness_algo: ClassVar[bool] = False

    def __init__(
        self,
        *,
        C: Optional[float] = None,
        kernel: Optional[KernelType] = None,
        seed: int = 888,
    ):
        kernel_name = f" ({kernel})" if kernel is not None else ""
        self.__name = f"SVM{kernel_name}"
        self.seed = seed
        self.C = SVC().C if C is None else C
        self.kernel = SVC().kernel if kernel is None else kernel
        self._hyperparameters = {"C": self.C, "kernel": self.kernel}

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


def select_svm(C: float, kernel: KernelType, seed: int) -> Union[LinearSVC, SVC]:
    """Select the appropriate SVM model for the given parameters."""
    random_state = np.random.RandomState(seed=seed)
    if kernel is KernelType.linear:
        return LinearSVC(C=C, dual=False, tol=1e-12, random_state=random_state)
    return SVC(C=C, kernel=str(kernel), gamma="auto", random_state=random_state)
