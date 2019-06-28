"""Kamiran and Calders 2012"""
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.utility.data_structures import PathTuple, TestPathTuple, DataTuple, TestTuple
from ethicml.implementations import kamiran

VALID_MODELS = {"LR", "SVM"}


class Kamiran(InAlgorithmAsync):
    """
    Kamiran and Calders 2012
    """

    def __init__(
        self, classifier: str = "LR", C: Optional[float] = None, kernel: Optional[str] = None
    ):
        super().__init__()
        if classifier not in VALID_MODELS:
            raise ValueError("results: classifier must be one of %r." % VALID_MODELS)
        self.classifier = classifier
        if C is None:
            if self.classifier == "LR":
                self.C = LogisticRegression().C
            elif self.classifier == "SVM":
                self.C = SVC().C
            else:
                raise NotImplementedError("Sever problem: this point should not be reached")
        else:
            self.C = C
        if kernel is None:
            if self.classifier == "SVM":
                self.kernel = SVC().kernel
            else:
                self.kernel = ""
        else:
            self.kernel = kernel

    def run(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        return kamiran.train_and_predict(
            train, test, classifier=self.classifier, C=self.C, kernel=self.kernel
        )

    def _script_command(
        self, train_paths: PathTuple, test_paths: TestPathTuple, pred_path: Path
    ) -> (List[str]):
        script = ["-m", kamiran.train_and_predict.__module__]
        args = conventional_interface(
            train_paths, test_paths, pred_path, self.classifier, str(self.C), self.kernel
        )
        return script + args

    @property
    def name(self) -> str:
        return f"Kamiran & Calders {self.classifier}"
