"""Wrapper for SKLearn implementation of SVM."""
from typing import Optional, Union

from sklearn.svm import SVC, LinearSVC
import pandas as pd

from ethicml.common import implements
from ethicml.utility import DataTuple, TestTuple
from .in_algorithm import InAlgorithm


class SVM(InAlgorithm):
    """Support Vector Machine."""

    def __init__(self, C: Optional[float] = None, kernel: Optional[str] = None):
        """Init SVM."""
        super().__init__()
        self.C = SVC().C if C is None else C
        self.kernel = SVC().kernel if kernel is None else kernel

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: Union[DataTuple, TestTuple]) -> pd.DataFrame:
        clf = select_svm(self.C, self.kernel)
        clf.fit(train.x, train.y.to_numpy().ravel())
        return pd.DataFrame(clf.predict(test.x), columns=["preds"])

    @property
    def name(self) -> str:
        """Getter for algorithm name."""
        return "SVM"


def select_svm(C: float, kernel: str) -> SVC:
    """Select the appropriate SVM model for the given parameters."""
    print(kernel)
    if kernel == "linear":
        return LinearSVC(C=C, dual=False, tol=1e-12, random_state=888)
    return SVC(C=C, kernel=kernel, gamma="auto", random_state=888)
