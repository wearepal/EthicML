"""Fairness without Demographics."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import List, TypedDict

from ranzen import implements

from ethicml.models.inprocess.in_subprocess import InAlgorithmSubprocess
from ethicml.utility import HyperParamType

__all__ = ["DRO"]


class DroArgs(TypedDict):
    """Args used in this module."""

    batch_size: int
    epochs: int
    eta: float
    network_size: list[int]


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

    @property
    @implements(InAlgorithmSubprocess)
    def hyperparameters(self) -> HyperParamType:
        _hyperparameters = asdict(self)
        _hyperparameters.pop("dir")  # this is not really a hyperparameter
        return _hyperparameters

    @property
    @implements(InAlgorithmSubprocess)
    def name(self) -> str:
        return "Dist Robust Optim"

    @implements(InAlgorithmSubprocess)
    def _get_path_to_script(self) -> list[str]:
        return ["-m", "ethicml.implementations.dro_tabular"]
