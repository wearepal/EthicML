"""Wrapper for SKLearn implementation of SVM."""
from pathlib import Path
from typing import List, Optional

from sklearn.svm import SVC

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync

from .shared import flag_interface

__all__ = ["SVMAsync"]


class SVMAsync(InAlgorithmAsync):
    """Support Vector Machine."""

    def __init__(self, C: Optional[float] = None, kernel: Optional[str] = None):
        super().__init__(name="SVM", is_fairness_algo=False)
        self.flags = {
            "c": SVC().C if C is None else C,
            "kernel": SVC().kernel if kernel is None else kernel,
        }

    def _script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(train_path, test_path, pred_path, self.flags)
        return ["-m", "ethicml.implementations.svm"] + args
