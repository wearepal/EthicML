"""Wrapper around Sci-Kit Learn Logistic Regression."""
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold

from ethicml.common import implements
from ethicml.utility import DataTuple, Prediction, SoftPrediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["LR", "LRCV", "LRProb"]


class LR(InAlgorithm):
    """Logistic regression with hard predictions."""

    def __init__(self, C: Optional[float] = None):
        """Init LR."""
        self.C = LogisticRegression().C if C is None else C
        super().__init__(name=f"Logistic Regression, C={self.C}", is_fairness_algo=False)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        clf = LogisticRegression(solver="liblinear", random_state=888, C=self.C, multi_class="auto")
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict(test.x)))


class LRProb(InAlgorithm):
    """Logistic regression with soft output."""

    def __init__(self, C: Optional[int] = None):
        """Init LRProb."""
        self.C = LogisticRegression().C if C is None else C
        super().__init__(name=f"Logistic Regression Prob, C={self.C}", is_fairness_algo=False)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> SoftPrediction:
        clf = LogisticRegression(solver="liblinear", random_state=888, C=self.C, multi_class="auto")
        clf.fit(train.x, train.y.to_numpy().ravel())
        return SoftPrediction(soft=pd.Series(clf.predict_proba(test.x)[:, 1]))


class LRCV(InAlgorithm):
    """Kind of a cheap hack for now, but gives a proper cross-valudeted LR."""

    def __init__(self) -> None:
        """Init LRCV."""
        super().__init__(name="LRCV", is_fairness_algo=False)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        folder = KFold(n_splits=3, shuffle=False)
        clf = LogisticRegressionCV(
            cv=folder, n_jobs=-1, random_state=888, solver="liblinear", multi_class="auto"
        )
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict(test.x)), info=dict(C=clf.C_[0]))
