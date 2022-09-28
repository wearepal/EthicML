"""Wrapper around Sci-Kit Learn Logistic Regression."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from ranzen import implements
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold

from ethicml.models.inprocess.in_algorithm import InAlgorithmDC
from ethicml.utility import DataTuple, Prediction, SoftPrediction, TestTuple

__all__ = ["LR", "LRCV"]


@dataclass
class LR(InAlgorithmDC):
    """Logistic regression with hard predictions.

    This is a wrapper around Sci-Kit Learn's LogisticRegression.
    See `the documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
    for details.

    :param C: The regularization parameter.
    """

    is_fairness_algo: ClassVar[bool] = False
    C: float = field(default_factory=lambda: LogisticRegression().C)

    @property
    @implements(InAlgorithmDC)
    def name(self) -> str:
        return f"Logistic Regression (C={self.C})"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple, seed: int = 888) -> LR:
        random_state = np.random.RandomState(seed=seed)
        self.clf = LogisticRegression(
            solver="liblinear", random_state=random_state, C=self.C, multi_class="auto"
        )
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.clf.predict(test.x)))

    @implements(InAlgorithmDC)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> SoftPrediction:
        random_state = np.random.RandomState(seed=seed)
        clf = LogisticRegression(
            solver="liblinear", random_state=random_state, C=self.C, multi_class="auto"
        )
        clf.fit(train.x, train.y.to_numpy().ravel())
        return SoftPrediction(soft=clf.predict_proba(test.x), info=self.hyperparameters)


@dataclass
class LRCV(InAlgorithmDC):
    """Kind of a cheap hack for now, but gives a proper cross-valudeted LR.

    :param n_splits: The number of splits for the cross-validation.
    """

    is_fairness_algo: ClassVar[bool] = False
    n_splits: int = 3

    @property
    @implements(InAlgorithmDC)
    def name(self) -> str:
        return "LRCV"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple, seed: int = 888) -> LRCV:
        random_state = np.random.RandomState(seed=seed)
        folder = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
        self.clf = LogisticRegressionCV(
            cv=folder, n_jobs=-1, random_state=random_state, solver="liblinear", multi_class="auto"
        )
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        params = self.hyperparameters
        params["C"] = self.clf.C_[0]
        return SoftPrediction(soft=self.clf.predict_proba(test.x), info=params)

    @implements(InAlgorithmDC)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        random_state = np.random.RandomState(seed=seed)
        folder = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
        clf = LogisticRegressionCV(
            cv=folder, n_jobs=-1, random_state=random_state, solver="liblinear", multi_class="auto"
        )
        clf.fit(train.x, train.y.to_numpy().ravel())
        params = self.hyperparameters
        params["C"] = clf.C_[0]
        return SoftPrediction(soft=clf.predict_proba(test.x), info=params)
