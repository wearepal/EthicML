"""Wrapper around Sci-Kit Learn Logistic Regression."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ranzen import implements
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold

from ethicml.utility import DataTuple, Prediction, SoftPrediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["LR", "LRCV", "LRProb"]


@dataclass
class LR(InAlgorithm):
    """Logistic regression with hard predictions."""

    C: float = field(default_factory=lambda: LogisticRegression().C)
    seed: int = 888
    is_fairness_algo = False

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return f"Logistic Regression (C={self.C})"

    @implements(InAlgorithm)
    def fit(self, train: DataTuple) -> InAlgorithm:
        random_state = np.random.RandomState(seed=self.seed)
        self.clf = LogisticRegression(
            solver="liblinear", random_state=random_state, C=self.C, multi_class="auto"
        )
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.clf.predict(test.x)))

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        random_state = np.random.RandomState(seed=self.seed)
        clf = LogisticRegression(
            solver="liblinear", random_state=random_state, C=self.C, multi_class="auto"
        )
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict(test.x)))


@dataclass
class LRProb(InAlgorithm):
    """Logistic regression with soft output."""

    C: float = field(default_factory=lambda: LogisticRegression().C)
    seed: int = 888
    is_fairness_algo = False

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return f"Logistic Regression Prob (C={self.C})"

    @implements(InAlgorithm)
    def fit(self, train: DataTuple) -> InAlgorithm:
        random_state = np.random.RandomState(seed=self.seed)
        self.clf = LogisticRegression(
            solver="liblinear", random_state=random_state, C=self.C, multi_class="auto"
        )
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return SoftPrediction(soft=pd.Series(self.clf.predict_proba(test.x)[:, 1]))

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> SoftPrediction:
        random_state = np.random.RandomState(seed=self.seed)
        clf = LogisticRegression(
            solver="liblinear", random_state=random_state, C=self.C, multi_class="auto"
        )
        clf.fit(train.x, train.y.to_numpy().ravel())
        return SoftPrediction(soft=pd.Series(clf.predict_proba(test.x)[:, 1]))


@dataclass
class LRCV(InAlgorithm):
    """Kind of a cheap hack for now, but gives a proper cross-valudeted LR."""

    n_splits: int = 3
    seed: int = 888
    is_fairness_algo = False

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "LRCV"

    @implements(InAlgorithm)
    def fit(self, train: DataTuple) -> InAlgorithm:
        random_state = np.random.RandomState(seed=self.seed)
        folder = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
        self.clf = LogisticRegressionCV(
            cv=folder, n_jobs=-1, random_state=random_state, solver="liblinear", multi_class="auto"
        )
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.clf.predict(test.x)), info=dict(C=self.clf.C_[0]))

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        random_state = np.random.RandomState(seed=self.seed)
        folder = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
        clf = LogisticRegressionCV(
            cv=folder, n_jobs=-1, random_state=random_state, solver="liblinear", multi_class="auto"
        )
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict(test.x)), info=dict(C=clf.C_[0]))
