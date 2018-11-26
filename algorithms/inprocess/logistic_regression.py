"""
Wrapper around Sci-Kit Learn Logistic Regression
"""

from typing import Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from algorithms.inprocess.in_algorithm import InAlgorithm


class LR(InAlgorithm):

    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        clf = LogisticRegression()
        clf.fit(train['x'], train['y'].values.ravel())
        return pd.DataFrame(clf.predict(test['x']), columns=["preds"])

    @property
    def name(self) -> str:
        return "Logistic Regression"
