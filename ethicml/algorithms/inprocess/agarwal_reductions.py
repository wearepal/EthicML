"""
implementation of Agarwal model
"""
from pathlib import Path
from typing import List

import pandas as pd

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.algorithms.utils import PathTuple, DataTuple
from ethicml.implementations import agarwal


VALID_FAIRNESS = {"DP", "EqOd"}
VALID_MODELS = {"LR", "SVM"}


class Agarwal(InAlgorithm):
    """
    Agarwal class
    """
    def __init__(self,
                 fairness: str = "DP",
                 classifier: str = "LR",
                 eps: float = 0.1,
                 iters=50,
                 hyperparams=None
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
        self.hyperparams = hyperparams

    def _run(self, train: DataTuple, test: DataTuple) -> pd.DataFrame:
        return agarwal.train_and_predict(train=train, test=test,
                                         classifier=self.classifier, fairness=self.fairness,
                                         eps=self.eps, iters=self.iters,
                                         hyperparams=self.hyperparams)

    def _script_command(self, train_paths: PathTuple,
                        test_paths: PathTuple, pred_path: Path) -> (List[str]):
        script = ['-m', agarwal.train_and_predict.__module__]
        args = conventional_interface(train_paths, test_paths, pred_path,
                                      str(self.classifier), str(self.fairness),
                                      str(self.eps), str(self.iters))
        return script + args

    @property
    def name(self) -> str:
        return f"Agarwal {self.classifier}"
