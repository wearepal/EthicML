"""Fairness without Demographics."""

from pathlib import Path
from typing import ClassVar, List, Optional, Union
from typing_extensions import TypedDict

from ranzen import implements

from .in_algorithm import InAlgoArgs, InAlgorithmAsync
from .shared import flag_interface

__all__ = ["DRO"]


class _Flags(TypedDict):
    batch_size: int
    epochs: int
    eta: float
    network_size: List[int]
    seed: int


class DroArgs(InAlgoArgs, _Flags):
    """Args used in this module."""


class DRO(InAlgorithmAsync):
    """Implementation of https://arxiv.org/abs/1806.08010 ."""

    is_fairness_algo: ClassVar[bool] = True

    def __init__(
        self,
        *,
        dir: Union[str, Path] = ".",
        eta: float = 0.5,
        epochs: int = 10,
        batch_size: int = 32,
        network_size: Optional[List[int]] = None,
        seed: int = 888,
    ):
        """Initialize the Distributionally Robust Optimization method.

        Args:
            dir: Directory to store the model.
            eta: Tolerance.
            epochs: The number of epochs to train for.
            batch_size: The batch size.
            network_size: The size of the network.
            seed: The seed for the random number generator.
        """
        self.seed = seed
        if network_size is None:
            network_size = [50]
        self.model_dir = dir if isinstance(dir, Path) else Path(dir)
        self.flags: _Flags = {
            "eta": eta,
            "batch_size": batch_size,
            "epochs": epochs,
            "network_size": network_size,
            "seed": seed,
        }
        self._hyperparameters = {
            "eta": eta,
            "epochs": epochs,
            "batch_size": batch_size,
            "network_size": f"{network_size}",
        }

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "Dist Robust Optim"

    @implements(InAlgorithmAsync)
    def _script_command(self, args: InAlgoArgs) -> List[str]:
        return ["-m", "ethicml.implementations.dro_tabular", flag_interface(self.flags, args)]
