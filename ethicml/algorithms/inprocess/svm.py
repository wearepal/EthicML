"""
Wrapper for SKLearn implementation of SVM
"""
from pathlib import Path
from typing import Optional, List, Union

from sklearn.svm import SVC
import pandas as pd

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.implementations import svm
from ethicml.utility.data_structures import DataTuple, TestTuple, PathTuple, TestPathTuple
from .shared import conventional_interface


class SVM(InAlgorithmAsync):
    """Support Vector Machine"""

    def __init__(self, C: Optional[float] = None, kernel: Optional[str] = None):
        super().__init__()
        self.C = SVC().C if C is None else C
        self.kernel = SVC().kernel if kernel is None else kernel

    def run(self, train: DataTuple, test: Union[DataTuple, TestTuple]) -> pd.DataFrame:
        return svm.train_and_predict(train, test, self.C, self.kernel)

    def _script_command(
        self, train_paths: PathTuple, test_paths: TestPathTuple, pred_path: Path
    ) -> List[str]:
        script = ["-m", svm.train_and_predict.__module__]
        args = conventional_interface(
            train_paths, test_paths, pred_path, str(self.C), str(self.kernel)
        )
        return script + args

    @property
    def name(self) -> str:
        return "SVM"
