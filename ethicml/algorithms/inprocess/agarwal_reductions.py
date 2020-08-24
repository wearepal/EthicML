"""Implementation of Agarwal model."""
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from ethicml.utility import ClassifierType, FairnessType

from .in_algorithm import InAlgorithmAsync
from .shared import flag_interface, settings_for_svm_lr

__all__ = ["Agarwal"]


VALID_FAIRNESS: Set[FairnessType] = {"DP", "EqOd"}
VALID_MODELS: Set[ClassifierType] = {"LR", "SVM"}


class Agarwal(InAlgorithmAsync):
    """Agarwal class."""

    def __init__(
        self,
        fairness: FairnessType = "DP",
        classifier: ClassifierType = "LR",
        eps: float = 0.1,
        iters: int = 50,
        C: Optional[float] = None,
        kernel: Optional[str] = None,
    ):
        if fairness not in VALID_FAIRNESS:
            raise ValueError("results: fairness must be one of %r." % VALID_FAIRNESS)
        if classifier not in VALID_MODELS:
            raise ValueError("results: classifier must be one of %r." % VALID_MODELS)
        super().__init__(name=f"Agarwal, {classifier}, {fairness}")
        chosen_c, chosen_kernel = settings_for_svm_lr(classifier, C, kernel)
        self.flags: Dict[str, Union[str, float, int]] = {
            "classifier": classifier,
            "fairness": fairness,
            "eps": eps,
            "iters": iters,
            "C": chosen_c,
            "kernel": chosen_kernel,
        }

    def _script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(train_path, test_path, pred_path, self.flags)
        return ["-m", "ethicml.implementations.agarwal"] + args
