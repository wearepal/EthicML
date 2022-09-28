"""Implementation of Agarwal model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TypedDict

from ranzen import implements

from ethicml.models.inprocess.in_subprocess import InAlgorithmSubprocess
from ethicml.models.inprocess.shared import settings_for_svm_lr
from ethicml.utility import ClassifierType, FairnessType, HyperParamType

__all__ = ["Agarwal"]

from ethicml.utility import KernelType

VALID_MODELS: set[ClassifierType] = {ClassifierType.lr, ClassifierType.svm}


class AgarwalArgs(TypedDict):
    """Args for the Agarwal implementation."""

    classifier: str
    fairness: str
    eps: float
    iters: int
    C: float
    kernel: str


@dataclass
class Agarwal(InAlgorithmSubprocess):
    """Agarwal class.

    A wrapper around the Exponentiated Gradient method documented
    `on this website <https://fairlearn.org/v0.7.0/api_reference/fairlearn.reductions.html#fairlearn.reductions.ExponentiatedGradient>`_.

    :param fairness: Type of fairness to enforce.
    :param classifier: Type of classifier to use.
    :param eps: Epsilon fo.
    :param iters: Number of iterations for the DP algorithm.
    :param C: C parameter for the SVM algorithm.
    :param kernel: Kernel type for the SVM algorithm.
    """

    fairness: FairnessType = FairnessType.dp
    classifier: ClassifierType = ClassifierType.lr
    eps: float = 0.1
    iters: int = 50
    C: Optional[float] = None
    kernel: Optional[KernelType] = None

    def __post_init__(self) -> None:
        assert self.fairness in (FairnessType.dp, FairnessType.eq_odds)

    @implements(InAlgorithmSubprocess)
    def _get_flags(self) -> AgarwalArgs:
        chosen_c, chosen_kernel = settings_for_svm_lr(self.classifier, self.C, self.kernel)
        # TODO: replace this with dataclasses.asdict()
        return {
            "classifier": self.classifier,
            "fairness": self.fairness,
            "eps": self.eps,
            "iters": self.iters,
            "C": chosen_c,
            "kernel": chosen_kernel if chosen_kernel is not None else "",
        }

    @property
    @implements(InAlgorithmSubprocess)
    def hyperparameters(self) -> HyperParamType:
        chosen_c, chosen_kernel = settings_for_svm_lr(self.classifier, self.C, self.kernel)
        _hyperparameters: HyperParamType = {
            "C": chosen_c,
            "iters": self.iters,
            "eps": self.eps,
            "fairness": self.fairness,
        }
        if self.classifier is ClassifierType.svm:
            assert chosen_kernel is not None
            _hyperparameters["kernel"] = chosen_kernel
        return _hyperparameters

    @property
    @implements(InAlgorithmSubprocess)
    def name(self) -> str:
        return f"Agarwal, {self.classifier}, {self.fairness}, {self.eps}"

    @implements(InAlgorithmSubprocess)
    def _get_path_to_script(self) -> list[str]:
        return ["-m", "ethicml.implementations.agarwal"]
