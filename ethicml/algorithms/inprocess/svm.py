"""
Wrapper for SKLearn implementation of SVM
"""
from typing import Optional

from sklearn.svm import SVC

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.implementations import svm


class SVM(InAlgorithmAsync):
    """Support Vector Machine"""

    def __init__(self, C: Optional[float] = None, kernel: Optional[str] = None):
        super().__init__()
        self.C = SVC().C if C is None else C
        self.kernel = SVC().kernel if kernel is None else kernel

    def run(self, train, test):
        return svm.train_and_predict(train, test, self.C, self.kernel)

    def _script_command(self, train_paths, test_paths, pred_path):
        script = ["-m", svm.train_and_predict.__module__]
        args = conventional_interface(
            train_paths, test_paths, pred_path, str(self.C), str(self.kernel)
        )
        return script + args

    @property
    def name(self) -> str:
        return "SVM"
