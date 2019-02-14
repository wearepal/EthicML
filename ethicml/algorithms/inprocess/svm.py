"""
Wrapper for SKLearn implementation of SVM
"""

from typing import Dict
import pandas as pd

from sklearn.svm import SVC
from .in_algorithm import InAlgorithm


class SVM(InAlgorithm):

    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        clf = SVC(random_state=888)
        clf.fit(train['x'], train['y'].values.ravel())
        return pd.DataFrame(clf.predict(test['x']), columns=["preds"])

    @property
    def name(self) -> str:
        return "SVM"
