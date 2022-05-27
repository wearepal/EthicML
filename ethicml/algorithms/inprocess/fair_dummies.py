"""Paper: "Achieving Equalized Odds by Resampling Sensitive Attributes," Y. Romano, S. Bates, and E. J. Candès, 2020.

https://github.com/yromano/fair_dummies
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from typing_extensions import Literal, TypedDict

from ranzen import implements

from ethicml.algorithms.inprocess.in_algorithm import HyperParamType
from ethicml.algorithms.inprocess.in_subprocess import InAlgorithmSubprocess

__all__ = ["FairDummies", "FairDummiesArgs"]


class FairDummiesArgs(TypedDict):
    """Args for the Fair Dummies implementation."""

    lr: float
    pretrain_pred_epochs: int
    pretrain_dis_epochs: int
    epochs: int
    loss_steps: int
    dis_steps: int
    batch_size: int
    model_type: Literal["deep_model", "linear_model"]
    lambda_vec: float
    second_moment_scaling: float


@dataclass
class FairDummies(InAlgorithmSubprocess):
    """FairDummie Method for enforcing Eq.Odds."""

    lr: float = 1e-3
    pretrain_pred_epochs: int = 2
    pretrain_dis_epochs: int = 2
    epochs: int = 50
    loss_steps: int = 1
    dis_steps: int = 1
    batch_size: int = 32
    model_type: str = "deep_model"
    lambda_vec: float = 0.9
    second_moment_scaling: float = 1e-5

    @implements(InAlgorithmSubprocess)
    def _get_path_to_script(self) -> List[str]:
        return ["-m", "ethicml.implementations.fair_dummies_romano"]

    @implements(InAlgorithmSubprocess)
    def _get_flags(self) -> FairDummiesArgs:
        model_type: Literal["deep_model", "linear_model"] = (
            "deep_model" if self.model_type.lower() == "deep_model" else "linear_model"
        )
        return {
            "lr": self.lr,
            "pretrain_pred_epochs": self.pretrain_pred_epochs,
            "pretrain_dis_epochs": self.pretrain_dis_epochs,
            "epochs": self.epochs,
            "loss_steps": self.loss_steps,
            "dis_steps": self.dis_steps,
            "batch_size": self.batch_size,
            "model_type": model_type,
            "lambda_vec": self.lambda_vec,
            "second_moment_scaling": self.second_moment_scaling,
        }

    @implements(InAlgorithmSubprocess)
    def get_hyperparameters(self) -> HyperParamType:
        return {
            "lr": self.lr,
            "pretrain_pred_epochs": self.pretrain_pred_epochs,
            "pretrain_dis_epochs": self.pretrain_dis_epochs,
            "epochs": self.epochs,
            "loss_steps": self.loss_steps,
            "dis_steps": self.dis_steps,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
            "lambda_vec": self.lambda_vec,
            "second_moment_scaling": self.second_moment_scaling,
        }

    @implements(InAlgorithmSubprocess)
    def get_name(self) -> str:
        return f"Fair Dummies {self.model_type}"
