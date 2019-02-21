"""
Wrapper around Sci-Kit Learn Logistic Regression
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from .in_algorithm import InAlgorithm
from ..utils import DataTuple


class LR(InAlgorithm):
    """Logistic regression with hard predictions"""
    def run(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        clf = LogisticRegression(random_state=888)
        clf.fit(train.x, train.y.values.ravel())
        return pd.DataFrame(clf.predict(test.x), columns=["preds"])

    @property
    def name(self) -> str:
        return "Logistic Regression"


class LRProb(InAlgorithm):
    """Logistic regression with soft output"""
    def run(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        clf = LogisticRegression(random_state=888)
        clf.fit(train.x, train.y.values.ravel())
        return pd.DataFrame(clf.predict_proba(test.x)[:, 1], columns=["preds"])

    @property
    def name(self) -> str:
        return "Logistic Regression- prob out"
