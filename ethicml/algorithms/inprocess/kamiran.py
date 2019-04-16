"""Kamiran and Calders 2012"""
from pathlib import Path
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.algorithms.utils import PathTuple
from ethicml.implementations import kamiran

VALID_MODELS = {"LR", "SVM"}


class Kamiran(InAlgorithm):
    """
    Kamiran and Calders 2012
    """

    def __init__(self, classifier: str = "LR", C: float = None, kernel: str = None):
        super().__init__()
        if classifier not in VALID_MODELS:
            raise ValueError("results: classifier must be one of %r." % VALID_MODELS)
        self.classifier = classifier
        self.C = C
        if self.C is None:
            if self.classifier == "LR":
                self.C = LogisticRegression().C
            elif self.classifier == "SVM":
                self.C = SVC().C
            else:
                raise NotImplementedError("Sever problem: this point should not be reached")
        self.kernel = kernel
        if self.kernel is None:
            if self.classifier == "SVM":
                self.kernel = SVC().kernel
            else:
                self.kernel = ""

    def _run(self, train, test):
        return kamiran.train_and_predict(train, test,
                                         classifier=self.classifier, C=self.C, kernel=self.kernel)

    def _script_command(self, train_paths: PathTuple, test_paths: PathTuple,
                        pred_path: Path) -> (List[str]):
        script = ['-m', kamiran.train_and_predict.__module__]
        args = conventional_interface(train_paths, test_paths, pred_path,
                                      self.classifier, str(self.C), self.kernel)
        return script + args

    @property
    def name(self) -> str:
        return f"Kamiran & Calders {self.classifier}"
