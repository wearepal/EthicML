"""
Wrapper around Sci-Kit Learn Logistic Regression
"""

from typing import Dict, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ethicml.algorithms.algorithm import Algorithm


class LR(Algorithm):

    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        clf = LogisticRegression()
        clf.fit(train['x'], train['y'].values.ravel())
        return pd.DataFrame(clf.predict(test['x']), columns=["preds"])

    def get_name(self) -> str:
        return "Logistic Regression"
