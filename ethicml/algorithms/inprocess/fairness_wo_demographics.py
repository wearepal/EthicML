"""Fairness without Demographics."""

from pathlib import Path
from typing import ClassVar, List, Optional
from typing_extensions import TypedDict

from ranzen import implements

from .in_algorithm import InAlgoArgs, InAlgorithm, InAlgorithmAsync
from .shared import flag_interface

__all__ = ["DRO"]

from ethicml.utility import DataTuple, Prediction, TestTuple


class DroArgs(TypedDict):
    """Args used in this module."""

    batch_size: int
    epochs: int
    eta: float
    network_size: List[int]
    seed: int


class DRO(InAlgorithmAsync):
    """Implementation of https://arxiv.org/abs/1806.08010 .

    :param dir: Directory to store the model.
    :param eta: Tolerance.
    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size.
    :param network_size: The size of the network.
    :param seed: The seed for the random number generator.
    """

    is_fairness_algo: ClassVar[bool] = True

    def __init__(
        self,
        *,
        dir: str = ".",
        eta: float = 0.5,
        epochs: int = 10,
        batch_size: int = 32,
        network_size: Optional[List[int]] = None,
        seed: int = 888,
    ):
        self.seed = seed
        if network_size is None:
            network_size = [50]
        self.model_dir = Path(dir)
        self.flags: DroArgs = {
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

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        return self._run(train, test)

    @implements(InAlgorithmAsync)
    def _script_command(self, in_algo_args: InAlgoArgs) -> List[str]:
        args = flag_interface(in_algo_args, self.flags)
        return ["-m", "ethicml.implementations.dro_tabular"] + args
