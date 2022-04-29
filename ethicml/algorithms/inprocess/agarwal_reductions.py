"""Implementation of Agarwal model."""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Set
from typing_extensions import TypedDict

from ranzen import implements

from ethicml.utility import ClassifierType, FairnessType

from .in_algorithm import InAlgoArgs, InAlgorithmAsync
from .shared import flag_interface, settings_for_svm_lr

__all__ = ["Agarwal"]

from ethicml.utility import KernelType

VALID_MODELS: Set[ClassifierType] = {ClassifierType.lr, ClassifierType.svm}


class AgarwalArgs(TypedDict):
    """Args for the Agarwal implementation."""

    classifier: str
    fairness: str
    eps: float
    iters: int
    C: float
    kernel: str
    seed: int


@dataclass
class Agarwal(InAlgorithmAsync):
    """Agarwal class.

    A wrapper around the Exponentiated Gradient method documented
    `on this website <https://fairlearn.org/v0.7.0/api_reference/fairlearn.reductions.html#fairlearn.reductions.ExponentiatedGradient>`_.

    :param dir: Directory to store the model.
    :param fairness: Type of fairness to enforce.
    :param classifier: Type of classifier to use.
    :param eps: Epsilon fo.
    :param iters: Number of iterations for the DP algorithm.
    :param C: C parameter for the SVM algorithm.
    :param kernel: Kernel type for the SVM algorithm.
    :param seed: Random seed.
    """

    is_fairness_algo: ClassVar[bool] = True
    dir: str = "."
    fairness: FairnessType = FairnessType.dp
    classifier: ClassifierType = ClassifierType.lr
    eps: float = 0.1
    iters: int = 50
    C: Optional[float] = None
    kernel: Optional[KernelType] = None
    seed: int = 888

    def __post_init__(self) -> None:
        self.model_dir = Path(self.dir)
        chosen_c, chosen_kernel = settings_for_svm_lr(self.classifier, self.C, self.kernel)
        assert self.fairness in (FairnessType.dp, FairnessType.eq_odds)
        self.flags: AgarwalArgs = {
            "classifier": str(self.classifier),
            "fairness": str(self.fairness),
            "eps": self.eps,
            "iters": self.iters,
            "C": chosen_c,
            "kernel": str(chosen_kernel) if chosen_kernel is not None else "",
            "seed": self.seed,
        }
        self._hyperparameters = {
            "C": chosen_c,
            "iters": self.iters,
            "eps": self.eps,
            "fairness": self.fairness,
        }
        if self.classifier is ClassifierType.svm:
            assert chosen_kernel is not None
            self._hyperparameters["kernel"] = chosen_kernel

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return f"Agarwal, {self.classifier}, {self.fairness}"

    @implements(InAlgorithmAsync)
    def _script_command(self, in_algo_args: InAlgoArgs) -> List[str]:
        return ["-m", "ethicml.implementations.agarwal"] + flag_interface(in_algo_args, self.flags)
