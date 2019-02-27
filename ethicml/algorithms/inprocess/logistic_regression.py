"""
Wrapper around Sci-Kit Learn Logistic Regression
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold

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


class LRCV(InAlgorithm):
    def run(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        folder = KFold(n_splits=3, random_state=888, shuffle=False)
        clf = LogisticRegressionCV(cv=folder, n_jobs=-1, random_state=888, solver='liblinear')
        clf.fit(train.x, train.y.values.ravel())
        return pd.DataFrame(clf.predict(test.x), columns=["preds"])

    @property
    def name(self) -> str:
        return "LRCV"
