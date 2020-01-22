"""Wrapper around Sci-Kit Learn Logistic Regression."""
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold

from ethicml.common import implements
from ethicml.utility.data_structures import DataTuple, TestTuple, Prediction
from .in_algorithm import InAlgorithm


class LR(InAlgorithm):
    """Logistic regression with hard predictions."""

    def __init__(self, C: Optional[float] = None):
        """Init LR."""
        super().__init__()
        self.C = LogisticRegression().C if C is None else C

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        clf = LogisticRegression(solver="liblinear", random_state=888, C=self.C, multi_class="auto")
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict(test.x)))

    @property
    def name(self) -> str:
        """Getter for algorithm name."""
        return f"Logistic Regression, C={self.C}"


class LRProb(InAlgorithm):
    """Logistic regression with soft output."""

    def __init__(self, C: Optional[int] = None):
        """Init LRProb."""
        super().__init__()
        self.C = LogisticRegression().C if C is None else C

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        clf = LogisticRegression(solver="liblinear", random_state=888, C=self.C, multi_class="auto")
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict_proba(test.x)[:, 1]))

    @property
    def name(self) -> str:
        """Getter for algorithm name."""
        return f"Logistic Regression Prob, C={self.C}"


class LRCV(InAlgorithm):
    """Kind of a cheap hack for now, but gives a proper cross-valudeted LR."""

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        folder = KFold(n_splits=3, random_state=888, shuffle=False)
        clf = LogisticRegressionCV(
            cv=folder, n_jobs=-1, random_state=888, solver="liblinear", multi_class="auto"
        )
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict(test.x)))

    @property
    def name(self) -> str:
        """Getter for algorithm name."""
        return "LRCV"
