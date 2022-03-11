"""Implementation of Agarwal model."""
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from ranzen import implements

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
        dir: Union[str, Path],
        fairness: FairnessType = "DP",
        classifier: ClassifierType = "LR",
        eps: float = 0.1,
        iters: int = 50,
        C: Optional[float] = None,
        kernel: Optional[str] = None,
        seed: int = 888,
    ):
        if fairness not in VALID_FAIRNESS:
            raise ValueError(f"results: fairness must be one of {VALID_FAIRNESS!r}.")
        if classifier not in VALID_MODELS:
            raise ValueError(f"results: classifier must be one of {VALID_MODELS!r}.")
        self.seed = seed
        self.is_fairness_algo = True
        self.model_dir = dir if isinstance(dir, Path) else Path(dir)
        chosen_c, chosen_kernel = settings_for_svm_lr(classifier, C, kernel)
        self.flags: Dict[str, Union[str, float, int]] = {
            "classifier": classifier,
            "fairness": fairness,
            "eps": eps,
            "iters": iters,
            "C": chosen_c,
            "kernel": chosen_kernel,
            "seed": seed,
        }

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return f"Agarwal, {self.flags['classifier']}, {self.flags['fairness']}"

    @implements(InAlgorithmAsync)
    def _run_script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(
            train_path=train_path, test_path=test_path, pred_path=pred_path, flags=self.flags
        )
        return ["-m", "ethicml.implementations.agarwal"] + args

    @implements(InAlgorithmAsync)
    def _fit_script_command(self, train_path: Path, model_path: Path) -> List[str]:
        args = flag_interface(train_path=train_path, model_path=model_path, flags=self.flags)
        return ["-m", "ethicml.implementations.agarwal"] + args

    @implements(InAlgorithmAsync)
    def _predict_script_command(
        self, model_path: Path, test_path: Path, pred_path: Path
    ) -> List[str]:
        args = flag_interface(
            model_path=model_path, test_path=test_path, pred_path=pred_path, flags=self.flags
        )
        return ["-m", "ethicml.implementations.agarwal"] + args
