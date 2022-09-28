"""HGR Classifier.

Paper: "Fairness-Aware Learning for Continuous Attributes and Treatments, J. Mary, C. Calauzènes, N. El Karoui, ICML 2019."
Based on: https://github.com/criteo-research/continuous-fairness
http://proceedings.mlr.press/v97/mary19a/mary19a.pdf
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TypedDict

from ranzen import implements

from ethicml.models.inprocess.in_subprocess import InAlgorithmSubprocess
from ethicml.utility import HyperParamType, ModelType

__all__ = ["HGR", "HgrArgs"]


class HgrArgs(TypedDict):
    """Args for the HGR implementation."""

    lr: float
    epochs: int
    mu: float
    batch_size: int
    model_type: str


@dataclass
class HGR(InAlgorithmSubprocess):
    """HGR Method."""

    lr: float = 1e-3
    epochs: int = 50
    mu: float = 0.98
    batch_size: int = 128
    model_type: ModelType = ModelType.deep

    @implements(InAlgorithmSubprocess)
    def _get_path_to_script(self) -> list[str]:
        return ["-m", "ethicml.implementations.hgr_method"]

    @implements(InAlgorithmSubprocess)
    def _get_flags(self) -> HgrArgs:
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "mu": self.mu,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
        }

    @property
    @implements(InAlgorithmSubprocess)
    def hyperparameters(self) -> HyperParamType:
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "mu": self.mu,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
        }

    @property
    @implements(InAlgorithmSubprocess)
    def name(self) -> str:
        return f"HGR {self.model_type}_model"
