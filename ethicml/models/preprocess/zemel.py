"""Zemel's Learned Fair Representations."""
from dataclasses import dataclass
from typing import List, Optional
from typing_extensions import TypedDict

from ranzen import implements

from .pre_subprocess import PreAlgorithmSubprocess

__all__ = ["Zemel"]


class ZemelArgs(TypedDict):
    """Arguments for the Zemel algorithm."""

    clusters: int
    Ax: float
    Ay: float
    Az: float
    max_iter: int
    maxfun: int
    epsilon: float
    threshold: float


@dataclass
class Zemel(PreAlgorithmSubprocess):
    """AIF360 implementation of Zemel's LFR."""

    threshold: float = 0.5
    clusters: int = 2
    Ax: float = 0.01
    Ay: float = 0.1
    Az: float = 0.5
    max_iter: int = 5_000
    maxfun: int = 5_000
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        self._in_size: Optional[int] = None  # the super class will set this for us

    @implements(PreAlgorithmSubprocess)
    def _get_flags(self) -> ZemelArgs:
        return {
            "clusters": self.clusters,
            "Ax": self.Ax,
            "Ay": self.Ay,
            "Az": self.Az,
            "max_iter": self.max_iter,
            "maxfun": self.maxfun,
            "epsilon": self.epsilon,
            "threshold": self.threshold,
        }

    @implements(PreAlgorithmSubprocess)
    def get_name(self) -> str:
        return "Zemel"

    @implements(PreAlgorithmSubprocess)
    def get_out_size(self) -> int:
        assert self._in_size is not None
        return self._in_size

    @implements(PreAlgorithmSubprocess)
    def _get_path_to_script(self) -> List[str]:
        return ["-m", "ethicml.implementations.zemel"]
