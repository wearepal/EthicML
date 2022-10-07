"""Paper: "Achieving Equalized Odds by Resampling Sensitive Attributes," Y. Romano, S. Bates, and E. J. Candès, 2020.

https://github.com/yromano/fair_dummies
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TypedDict

from ranzen import implements

from ethicml.models.inprocess.in_subprocess import InAlgorithmSubprocess
from ethicml.utility.data_structures import HyperParamType, ModelType

__all__ = ["AdvDebiasing", "AdvDebArgs"]


class AdvDebArgs(TypedDict):
    """Args for the Agarwal implementation."""

    lr: float
    n_clf_epochs: int
    n_adv_epochs: int
    n_epoch_combined: int
    batch_size: int
    model_type: str
    lambda_vec: float


@dataclass
class AdvDebiasing(InAlgorithmSubprocess):
    """Adversarial Debiasing Method."""

    lr = 0.5
    n_clf_epochs = 2
    n_adv_epochs = 2
    n_epoch_combined = 40
    batch_size = 32
    model_type: ModelType = ModelType.deep
    lambda_vec = 0.999999

    @implements(InAlgorithmSubprocess)
    def _get_path_to_script(self) -> list[str]:
        return ["-m", "ethicml.implementations.adv_debiasing_method"]

    @implements(InAlgorithmSubprocess)
    def _get_flags(self) -> AdvDebArgs:
        return {
            "lr": self.lr,
            "n_clf_epochs": self.n_clf_epochs,
            "n_adv_epochs": self.n_adv_epochs,
            "n_epoch_combined": self.n_epoch_combined,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
            "lambda_vec": self.lambda_vec,
        }

    @property
    @implements(InAlgorithmSubprocess)
    def hyperparameters(self) -> HyperParamType:
        return {
            "lr": self.lr,
            "n_clf_epochs": self.n_clf_epochs,
            "n_adv_epochs": self.n_adv_epochs,
            "n_epoch_combined": self.n_epoch_combined,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
            "lambda_vec": self.lambda_vec,
        }

    @property
    @implements(InAlgorithmSubprocess)
    def name(self) -> str:
        return "Adversarial Debiasing"
