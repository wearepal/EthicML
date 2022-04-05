"""Beutel's algorithm."""
from pathlib import Path
from typing import List, Sequence, Union
from typing_extensions import TypedDict

from ranzen import implements

from ethicml.utility import FairnessType

from .interface import flag_interface
from .pre_algorithm import PreAlgoArgs, PreAlgorithmAsync

__all__ = ["Beutel"]


class BeutelArgs(TypedDict):
    """Args for the Beutel Implementation."""

    fairness: FairnessType
    enc_size: List[int]
    adv_size: List[int]
    pred_size: List[int]
    enc_activation: str
    adv_activation: str
    batch_size: int
    y_loss: str
    s_loss: str
    epochs: int
    adv_weight: float
    validation_pcnt: float
    seed: int


class Beutel(PreAlgorithmAsync):
    """Beutel's adversarially learned fair representations."""

    def __init__(
        self,
        fairness: FairnessType = "DP",
        *,
        dir: Union[str, Path] = ".",
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
        self.flags: BeutelArgs = {
            "fairness": fairness,
            "enc_size": list(enc_size),
            "adv_size": list(adv_size),
            "pred_size": list(pred_size),
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
    def _script_command(self, pre_algo_args: PreAlgoArgs) -> List[str]:
        return ["-m", "ethicml.implementations.beutel"] + flag_interface(pre_algo_args, self.flags)
