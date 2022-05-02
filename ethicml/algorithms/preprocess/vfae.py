"""Variational Fair Auto-Encoder by Louizos et al."""
from dataclasses import dataclass, field
from typing import List
from typing_extensions import TypedDict

from ranzen import implements

from ethicml.algorithms.preprocess.pre_subprocess import PreAlgorithmSubprocess

__all__ = ["VFAE"]

from ethicml.utility import FairnessType


class VfaeArgs(TypedDict):
    """Args object of VFAE."""

    dataset: str
    supervised: bool
    epochs: int
    batch_size: int
    fairness: str
    latent_dims: int
    z1_enc_size: List[int]
    z2_enc_size: List[int]
    z1_dec_size: List[int]


@dataclass
class VFAE(PreAlgorithmSubprocess):
    """VFAE Object - see implementation file for details."""

    dataset: str = "toy"
    supervised: bool = True
    epochs: int = 10
    batch_size: int = 32
    fairness: FairnessType = FairnessType.dp
    latent_dims: int = 50
    z1_enc_size: List[int] = field(default_factory=lambda: [100])
    z2_enc_size: List[int] = field(default_factory=lambda: [100])
    z1_dec_size: List[int] = field(default_factory=lambda: [100])

    @implements(PreAlgorithmSubprocess)
    def _get_flags(self) -> VfaeArgs:
        # TODO: replace this with dataclasses.asdict()
        return {
            "supervised": self.supervised,
            "fairness": str(self.fairness),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "dataset": self.dataset,
            "latent_dims": self.latent_dims,
            "z1_enc_size": self.z1_enc_size,
            "z2_enc_size": self.z2_enc_size,
            "z1_dec_size": self.z1_dec_size,
        }

    @implements(PreAlgorithmSubprocess)
    def get_out_size(self) -> int:
        return self.latent_dims

    @implements(PreAlgorithmSubprocess)
    def get_name(self) -> str:
        return "VFAE"

    @implements(PreAlgorithmSubprocess)
    def _get_path_to_script(self) -> List[str]:
        return ["-m", "ethicml.implementations.vfae"]
