"""
Logistic regaression with soft output
"""
from typing import Optional

from sklearn.linear_model import LogisticRegression

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.implementations import logistic_regression_probability


class LRProb(InAlgorithm):
    """Logistic regression with soft output"""
    def __init__(self, C: Optional[int] = None):
        super().__init__()
        self.C = LogisticRegression().C if C is None else C

    def _run(self, train, test):
        return logistic_regression_probability.train_and_predict(train, test, self.C)

    def _script_command(self, train_paths, test_paths, pred_path):
        script = ['-m', logistic_regression_probability.train_and_predict.__module__]
        args = conventional_interface(train_paths, test_paths, pred_path, str(self.C))
        return script + args

    @property
    def name(self) -> str:
        return "Logistic Regression Prob"
