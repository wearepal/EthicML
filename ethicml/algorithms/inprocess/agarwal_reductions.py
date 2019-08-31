"""
implementation of Agarwal model
"""
from pathlib import Path
from typing import List

import pandas as pd

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.utility.data_structures import PathTuple, TestPathTuple, DataTuple, TestTuple
from ethicml.implementations import agarwal
from ._shared import conventional_interface, settings_for_svm_lr


VALID_FAIRNESS = {"DP", "EqOd"}
VALID_MODELS = {"LR", "SVM"}


class Agarwal(InAlgorithmAsync):
    """
    Agarwal class
    """

    def __init__(
        self,
        fairness: str = "DP",
        classifier: str = "LR",
        eps: float = 0.1,
        iters=50,
        C=None,
        kernel=None,
    ):
        if fairness not in VALID_FAIRNESS:
            raise ValueError("results: fairness must be one of %r." % VALID_FAIRNESS)
        if classifier not in VALID_MODELS:
            raise ValueError("results: classifier must be one of %r." % VALID_MODELS)
        super().__init__()
        self.classifier = classifier
        self.fairness = fairness
        self.eps = eps
        self.iters = iters
        self.C, self.kernel = settings_for_svm_lr(classifier, C, kernel)

    def run(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        return agarwal.train_and_predict(
            train=train,
            test=test,
            classifier=self.classifier,
            fairness=self.fairness,
            eps=self.eps,
            iters=self.iters,
            C=self.C,
            kernel=self.kernel,
        )

    def _script_command(
        self, train_paths: PathTuple, test_paths: TestPathTuple, pred_path: Path
    ) -> (List[str]):
        script = ["-m", agarwal.train_and_predict.__module__]
        args = conventional_interface(
            train_paths,
            test_paths,
            pred_path,
            str(self.classifier),
            str(self.fairness),
            str(self.eps),
            str(self.iters),
            str(self.C),
            self.kernel,
        )
        return script + args

    @property
    def name(self) -> str:
        return f"Agarwal {self.classifier}"
