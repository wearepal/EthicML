"""Beutel's algorithm."""
from pathlib import Path
from typing import Dict, List, Sequence, Union

from ranzen import implements

from ethicml.utility import FairnessType

from .interface import flag_interface
from .pre_algorithm import PreAlgorithmAsync

__all__ = ["Beutel"]


class Beutel(PreAlgorithmAsync):
    """Beutel's adversarially learned fair representations."""

    def __init__(
        self,
        dir: Union[str, Path],
        fairness: FairnessType = "DP",
        *,
        enc_size: Sequence[int] = (40,),
        adv_size: Sequence[int] = (40,),
        pred_size: Sequence[int] = (40,),
        enc_activation: str = "Sigmoid()",
        adv_activation: str = "Sigmoid()",
        batch_size: int = 64,
        y_loss: str = "BCELoss()",
        s_loss: str = "BCELoss()",
        epochs: int = 50,
        adv_weight: float = 1.0,
        validation_pcnt: float = 0.1,
        seed: int = 888,
    ):
        self.seed = seed
        self._out_size = enc_size[-1]
        self.model_dir = dir if isinstance(dir, Path) else Path(dir)
        self.flags: Dict[str, Union[str, Sequence[int], int, float]] = {
            "fairness": fairness,
            "enc_size": enc_size,
            "adv_size": adv_size,
            "pred_size": pred_size,
            "enc_activation": enc_activation,
            "adv_activation": adv_activation,
            "batch_size": batch_size,
            "y_loss": y_loss,
            "s_loss": s_loss,
            "epochs": epochs,
            "adv_weight": adv_weight,
            "validation_pcnt": validation_pcnt,
            "seed": seed,
        }

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return f"Beutel {self.flags['fairness']}"

    @implements(PreAlgorithmAsync)
    def _run_script_command(
        self, train_path: Path, test_path: Path, new_train_path: Path, new_test_path: Path
    ) -> List[str]:
        args = flag_interface(
            train_path=train_path,
            test_path=test_path,
            new_train_path=new_train_path,
            new_test_path=new_test_path,
            flags=self.flags,
        )
        return ["-m", "ethicml.implementations.beutel"] + args

    @implements(PreAlgorithmAsync)
    def _fit_script_command(
        self, train_path: Path, new_train_path: Path, model_path: Path
    ) -> List[str]:
        args = flag_interface(
            train_path=train_path,
            new_train_path=new_train_path,
            model_path=model_path,
            flags=self.flags,
        )
        return ["-m", "ethicml.implementations.beutel"] + args

    @implements(PreAlgorithmAsync)
    def _transform_script_command(
        self, model_path: Path, test_path: Path, new_test_path: Path
    ) -> List[str]:
        args = flag_interface(
            model_path=model_path,
            test_path=test_path,
            new_test_path=new_test_path,
            flags=self.flags,
        )
        return ["-m", "ethicml.implementations.beutel"] + args
