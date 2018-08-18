"""
Wrapper for SKLearn implementation of SVM
"""

from typing import Dict
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from ethicml.algorithms.algorithm import Algorithm


class SVM(Algorithm):

    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> np.array:
        clf = SVC()
        clf.fit(train['x'], train['y'].values.ravel())
        return clf.predict(test['x'])

    def get_name(self) -> str:
        return "SVM"
