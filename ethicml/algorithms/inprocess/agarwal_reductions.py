"""Implementation of Agarwal model."""
from pathlib import Path
from typing import Dict, List, Optional, Union

from kit import parsable

from ethicml.utility import ClassifierType, FairnessType, str_to_enum

from .in_algorithm import InAlgorithmAsync
from .shared import SVMKernel, flag_interface, settings_for_svm_lr

__all__ = ["Agarwal"]


class Agarwal(InAlgorithmAsync):
    """Agarwal class."""

    @parsable
    def __init__(
        self,
        fairness: Union[str, FairnessType] = FairnessType.DP,
        classifier: Union[str, ClassifierType] = ClassifierType.LR,
        eps: float = 0.1,
        iters: int = 50,
        C: Optional[float] = None,
        kernel: Optional[Union[str, SVMKernel]] = None,
    ):
        super().__init__(name=f"Agarwal, {str(classifier).upper()}, {fairness}")
        fairness = str_to_enum(fairness, enum=FairnessType)
        classifier = str_to_enum(classifier, enum=ClassifierType)
        kernel = None if kernel is None else str_to_enum(kernel.upper(), enum=SVMKernel)
        chosen_c, chosen_kernel = settings_for_svm_lr(classifier, C, kernel)
        self.flags: Dict[str, Union[str, float, int]] = {
            "classifier": str(classifier),
            "fairness": str(fairness),
            "eps": eps,
            "iters": iters,
            "C": chosen_c,
            "kernel": str(chosen_kernel),
        }

    def _script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(train_path, test_path, pred_path, self.flags)
        return ["-m", "ethicml.implementations.agarwal"] + args
