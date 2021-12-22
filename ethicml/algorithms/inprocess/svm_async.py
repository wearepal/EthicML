"""Wrapper for SKLearn implementation of SVM."""
from pathlib import Path
from typing import List, Optional, Union

from ranzen import implements
from sklearn.svm import SVC

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync

from .shared import flag_interface

__all__ = ["SVMAsync"]


class SVMAsync(InAlgorithmAsync):
    """Support Vector Machine."""

    def __init__(
        self,
        dir: Union[str, Path],
        C: Optional[float] = None,
        kernel: Optional[str] = None,
        seed: int = 888,
    ):
        super().__init__(name="SVM", is_fairness_algo=False, seed=seed)
        self.model_dir = dir if isinstance(dir, Path) else Path(dir)
        self.flags = {
            "c": SVC().C if C is None else C,
            "kernel": SVC().kernel if kernel is None else kernel,
            "seed": seed,
        }

    @implements(InAlgorithmAsync)
    def _run_script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(
            train_path=train_path, test_path=test_path, pred_path=pred_path, flags=self.flags
        )
        return ["-m", "ethicml.implementations.svm"] + args

    @implements(InAlgorithmAsync)
    def _fit_script_command(self, train_path: Path, model_path: Path) -> List[str]:
        args = flag_interface(train_path=train_path, model_path=model_path, flags=self.flags)
        return ["-m", "ethicml.implementations.svm"] + args

    @implements(InAlgorithmAsync)
    def _predict_script_command(
        self, model_path: Path, test_path: Path, pred_path: Path
    ) -> List[str]:
        args = flag_interface(
            model_path=model_path, test_path=test_path, pred_path=pred_path, flags=self.flags
        )
        return ["-m", "ethicml.implementations.svm"] + args
