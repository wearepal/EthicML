"""
Wrapper around Sci-Kit Learn Logistic Regression
"""
from typing import Optional

from sklearn.linear_model import LogisticRegression

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.implementations import (
    logistic_regression,
    logistic_regression_probability,
    logistic_regression_cross_validated,
)


class LR(InAlgorithmAsync):
    """Logistic regression with hard predictions"""

    def __init__(self, C: Optional[float] = None):
        super().__init__()
        self.C = LogisticRegression().C if C is None else C

    def run(self, train, test):
        return logistic_regression.train_and_predict(train, test, self.C)

    def _script_command(self, train_paths, test_paths, pred_path):
        script = ["-m", logistic_regression.train_and_predict.__module__]
        args = conventional_interface(train_paths, test_paths, pred_path, str(self.C))
        return script + args

    @property
    def name(self) -> str:
        return "Logistic Regression"


class LRProb(InAlgorithmAsync):
    """Logistic regression with soft output"""

    def __init__(self, C: Optional[int] = None):
        super().__init__()
        self.C = LogisticRegression().C if C is None else C

    def run(self, train, test):
        return logistic_regression_probability.train_and_predict(train, test, self.C)

    def _script_command(self, train_paths, test_paths, pred_path):
        script = ["-m", logistic_regression_probability.train_and_predict.__module__]
        args = conventional_interface(train_paths, test_paths, pred_path, str(self.C))
        return script + args

    @property
    def name(self) -> str:
        return "Logistic Regression Prob"


class LRCV(InAlgorithmAsync):
    """Kind of a cheap hack for now, but gives a proper cross-valudeted LR"""

    def run(self, train, test):
        return logistic_regression_cross_validated.train_and_predict(train, test)

    def _script_command(self, train_paths, test_paths, pred_path):
        script = ["-m", logistic_regression_cross_validated.train_and_predict.__module__]
        args = conventional_interface(train_paths, test_paths, pred_path)
        return script + args

    @property
    def name(self) -> str:
        return "LRCV"
