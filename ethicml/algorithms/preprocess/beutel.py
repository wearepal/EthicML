"""Beutel's algorithm."""
from dataclasses import dataclass, field
from typing import List
from typing_extensions import TypedDict

from ranzen import implements

from ethicml.utility import FairnessType

from .pre_subprocess import PreAlgorithmSubprocess

__all__ = ["Beutel"]


class BeutelArgs(TypedDict):
    """Args for the Beutel Implementation."""

    fairness: str
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


@dataclass
class Beutel(PreAlgorithmSubprocess):
    """Beutel's adversarially learned fair representations."""

    fairness: FairnessType = FairnessType.dp
    enc_size: List[int] = field(default_factory=lambda: [40])
    adv_size: List[int] = field(default_factory=lambda: [40])
    pred_size: List[int] = field(default_factory=lambda: [40])
    enc_activation: str = "Sigmoid()"
    adv_activation: str = "Sigmoid()"
    batch_size: int = 64
    y_loss: str = "CrossEntropyLoss()"
    s_loss: str = "BCELoss()"
    epochs: int = 50
    adv_weight: float = 1.0
    validation_pcnt: float = 0.1

    @implements(PreAlgorithmSubprocess)
    def get_out_size(self) -> int:
        return self.enc_size[-1]

    @implements(PreAlgorithmSubprocess)
    def _get_flags(self) -> BeutelArgs:
        # TODO: replace this with dataclasses.asdict()
        return {
            "fairness": str(self.fairness),
            "enc_size": list(self.enc_size),
            "adv_size": list(self.adv_size),
            "pred_size": list(self.pred_size),
            "enc_activation": self.enc_activation,
            "adv_activation": self.adv_activation,
            "batch_size": self.batch_size,
            "y_loss": self.y_loss,
            "s_loss": self.s_loss,
            "epochs": self.epochs,
            "adv_weight": self.adv_weight,
            "validation_pcnt": self.validation_pcnt,
        }

    @implements(PreAlgorithmSubprocess)
    def get_name(self) -> str:
        return f"Beutel {self.fairness}"

    @implements(PreAlgorithmSubprocess)
    def _get_path_to_script(self) -> List[str]:
        return ["-m", "ethicml.implementations.beutel"]
