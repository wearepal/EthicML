"""Implementation of Agarwal model."""
from pathlib import Path
from typing import ClassVar, List, Optional, Set, Union
from typing_extensions import TypedDict

from ranzen import implements, parsable

from ethicml.utility import ClassifierType, FairnessType

from .in_algorithm import InAlgoArgs, InAlgorithmAsync
from .shared import flag_interface, settings_for_svm_lr

__all__ = ["Agarwal"]

from ...utility import KernelType

VALID_FAIRNESS: Set[FairnessType] = {FairnessType.dp, FairnessType.eqod}
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


class Agarwal(InAlgorithmAsync):
    """Agarwal class.

    A wrapper around the Exponentiated Gradient method documented `here <https://fairlearn.org/v0.7.0/api_reference/fairlearn.reductions.html#fairlearn.reductions.ExponentiatedGradient>`_.
    """

    is_fairness_algo: ClassVar[bool] = True

    @parsable
    def __init__(
        self,
        *,
        dir: Union[str, Path] = ".",
        fairness: FairnessType = FairnessType.dp,
        classifier: ClassifierType = ClassifierType.lr,
        eps: float = 0.1,
        iters: int = 50,
        C: Optional[float] = None,
        kernel: Optional[KernelType] = None,
        seed: int = 888,
    ):
        """Initialize the Agarwal algorithm.

        Args:
            dir: Directory to store the model.
            fairness: Type of fairness to enforce.
            classifier: Type of classifier to use.
            eps: Epsilon fo.
            iters: Number of iterations for the DP algorithm.
            C: C parameter for the SVM algorithm.
            kernel: Kernel type for the SVM algorithm.
            seed: Random seed.
        """
        self.seed = seed
        self.model_dir = dir if isinstance(dir, Path) else Path(dir)
        chosen_c, chosen_kernel = settings_for_svm_lr(classifier, C, kernel)
        self.flags: AgarwalArgs = {
            "classifier": str(classifier),
            "fairness": str(fairness),
            "eps": eps,
            "iters": iters,
            "C": chosen_c,
            "kernel": str(chosen_kernel),
            "seed": seed,
        }
        self._hyperparameters = {"C": chosen_c, "iters": iters, "eps": eps, "fairness": fairness}
        if classifier == ClassifierType.svm:
            self._hyperparameters["kernel"] = chosen_kernel

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return f"Agarwal, {self.flags['classifier']}, {self.flags['fairness']}"

    @implements(InAlgorithmAsync)
    def _script_command(self, in_algo_args: InAlgoArgs) -> List[str]:
        return ["-m", "ethicml.implementations.agarwal"] + flag_interface(in_algo_args, self.flags)
