"""Wrapper for SKLearn implementation of SVM."""
from dataclasses import dataclass, field
from typing import ClassVar, Union
from typing_extensions import Literal, TypeAlias

import numpy as np
import pandas as pd
from ranzen import implements
from sklearn.svm import SVC, LinearSVC

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["SVM"]

KernelType: TypeAlias = Literal["linear", "rbf", "poly", "sigmoid"]


@dataclass
class SVM(InAlgorithm):
    """A wraper around the SciKitLearn Support Vector Classifier (SVC) model.

    Documentation for the underlying classifier can be found `here <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """

    C: float = field(default_factory=lambda: SVC().C)
    kernel: KernelType = field(default_factory=lambda: SVC().kernel)
    seed: int = 888
    is_fairness_algo: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self._hyperparameters = {"C": self.C, "kernel": self.kernel}

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return f"SVM ({self.kernel})"

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
