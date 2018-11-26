"""
Wrapper for SKLearn implementation of SVM
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from ethicml.algorithms.algorithm import Algorithm


class SVM(Algorithm):

    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        clf = SVC(kernel='linear')
        clf.fit(train['x'], train['y'].values.ravel())
        return pd.DataFrame(clf.predict(test['x']), columns=["preds"])

    def get_name(self) -> str:
        return "SVM"
