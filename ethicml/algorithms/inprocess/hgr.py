"""HGR Classifier.

Paper: "Fairness-Aware Learning for Continuous Attributes and Treatments, J. Mary, C. Calauzènes, N. El Karoui, ICML 2019."
Based on: https://github.com/criteo-research/continuous-fairness
http://proceedings.mlr.press/v97/mary19a/mary19a.pdf
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from typing_extensions import Literal, TypedDict

from ranzen import implements

from ethicml.algorithms.inprocess.in_subprocess import InAlgorithmSubprocess
from ethicml.utility import HyperParamType

__all__ = ["HGR", "HgrArgs"]


class HgrArgs(TypedDict):
    """Args for the HGR implementation."""

    lr: float
    epochs: int
    mu: float
    batch_size: int
    model_type: Literal["deep_model", "linear_model"]


@dataclass
class HGR(InAlgorithmSubprocess):
    """HGR Method."""

    lr: float = 1e-3
    epochs: int = 50
    mu: float = 0.98
    batch_size: int = 128
    model_type: str = "deep_model"

    @implements(InAlgorithmSubprocess)
    def _get_path_to_script(self) -> List[str]:
        return ["-m", "ethicml.implementations.hgr_method"]

    @implements(InAlgorithmSubprocess)
    def _get_flags(self) -> HgrArgs:
        model_type: Literal["deep_model", "linear_model"] = (
            "deep_model" if self.model_type.lower() == "deep_model" else "linear_model"
        )
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "mu": self.mu,
            "batch_size": self.batch_size,
            "model_type": model_type,
        }

    @implements(InAlgorithmSubprocess)
    def get_hyperparameters(self) -> HyperParamType:
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "mu": self.mu,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
        }

    @implements(InAlgorithmSubprocess)
    def get_name(self) -> str:
        return f"HGR {self.model_type}"
