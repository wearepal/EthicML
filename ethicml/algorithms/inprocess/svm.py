"""
Wrapper for SKLearn implementation of SVM
"""

import pandas as pd

from sklearn.svm import SVC
from .in_algorithm import InAlgorithm
from ..utils import DataTuple


class SVM(InAlgorithm):
    """Support Vector Machine"""
    def run(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        clf = SVC(gamma='auto', random_state=888)
        clf.fit(train.x, train.y.values.ravel())
        return pd.DataFrame(clf.predict(test.x), columns=["preds"])

    @property
    def name(self) -> str:
        return "SVM"
