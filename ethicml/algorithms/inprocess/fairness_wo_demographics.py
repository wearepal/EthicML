"""Fairness without Demographics."""

from dataclasses import asdict, dataclass, field
from typing import List
from typing_extensions import TypedDict

from ranzen import implements

from ethicml.algorithms.inprocess.in_subprocess import InAlgorithmSubprocess
from ethicml.utility import HyperParamType

__all__ = ["DRO"]


class DroArgs(TypedDict):
    """Args used in this module."""

    batch_size: int
    epochs: int
    eta: float
    network_size: List[int]


@dataclass
class DRO(InAlgorithmSubprocess):
    """Implementation of https://arxiv.org/abs/1806.08010 .

    :param eta: Tolerance.
    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size.
    :param network_size: The size of the network.
    """

    eta: float = 0.5
    epochs: int = 10
    batch_size: int = 32
    network_size: List[int] = field(default_factory=lambda: [50])

    @implements(InAlgorithmSubprocess)
    def _get_flags(self) -> DroArgs:
        # TODO: replace this with dataclasses.asdict()
        return {
            "eta": self.eta,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "network_size": self.network_size,
        }

    @implements(InAlgorithmSubprocess)
    def get_hyperparameters(self) -> HyperParamType:
        _hyperparameters = asdict(self)
        _hyperparameters.pop("dir")  # this is not really a hyperparameter
        return _hyperparameters

    @implements(InAlgorithmSubprocess)
    def get_name(self) -> str:
        return "Dist Robust Optim"

    @implements(InAlgorithmSubprocess)
    def _get_path_to_script(self) -> List[str]:
        return ["-m", "ethicml.implementations.dro_tabular"]
