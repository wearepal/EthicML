"""Variational Fair Auto-Encoder by Louizos et al."""
from pathlib import Path
from typing import List, Optional, Union
from typing_extensions import TypedDict

from ranzen import implements

from .interface import flag_interface
from .pre_algorithm import PreAlgoArgs, PreAlgorithmAsync

__all__ = ["VFAE"]


class VfaeArgs(TypedDict):
    """Args object of VFAE."""

    supervised: bool
    fairness: str
    batch_size: int
    epochs: int
    dataset: str
    latent_dims: int
    z1_enc_size: List[int]
    z2_enc_size: List[int]
    z1_dec_size: List[int]
    seed: int


class VFAE(PreAlgorithmAsync):
    """VFAE Object - see implementation file for details."""

    def __init__(
        self,
        dataset: str,
        *,
        dir: Union[str, Path] = ".",
        supervised: bool = True,
        epochs: int = 10,
        batch_size: int = 32,
        fairness: str = "DI",
        latent_dims: int = 50,
        z1_enc_size: Optional[List[int]] = None,
        z2_enc_size: Optional[List[int]] = None,
        z1_dec_size: Optional[List[int]] = None,
        seed: int = 888,
    ):
        self.seed = seed
        self._out_size = latent_dims
        self.model_dir = dir if isinstance(dir, Path) else Path(dir)

        if z1_enc_size is None:
            z1_enc_size = [100]
        if z2_enc_size is None:
            z2_enc_size = [100]
        if z1_dec_size is None:
            z1_dec_size = [100]

        self.flags: VfaeArgs = {
            "supervised": supervised,
            "fairness": fairness,
            "batch_size": batch_size,
            "epochs": epochs,
            "dataset": dataset,
            "latent_dims": latent_dims,
            "z1_enc_size": z1_enc_size,
            "z2_enc_size": z2_enc_size,
            "z1_dec_size": z1_dec_size,
            "seed": seed,
        }

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "VFAE"

    @implements(PreAlgorithmAsync)
    def _script_command(self, pre_algo_args: PreAlgoArgs) -> List[str]:
        return ["-m", "ethicml.implementations.vfae"] + flag_interface(pre_algo_args, self.flags)
