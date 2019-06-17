"""Beutel's algorithm"""
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

from .pre_algorithm import PreAlgorithmAsync
from .interface import flag_interface
from ..utils import DataTuple, TestTuple, PathTuple, TestPathTuple


class Beutel(PreAlgorithmAsync):
    """Beutel's adversarially learned fair representations"""

    def __init__(
        self,
        fairness: str = "DI",
        enc_size: Sequence[int] = (40,),
        adv_size: Sequence[int] = (40,),
        pred_size: Sequence[int] = (40,),
        enc_activation: str = "Sigmoid()",
        adv_activation: str = "Sigmoid()",
        batch_size: int = 64,
        y_loss: str = "BCELoss()",
        s_loss: str = "BCELoss()",
        epochs: int = 50,
        adv_weight: float = 1.0,
        validation_pcnt: float = 0.1,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.fairness = fairness
        self.enc_size = enc_size
        self.adv_size = adv_size
        self.pred_size = pred_size
        self.enc_activation = enc_activation
        self.adv_activation = adv_activation
        self.batch_size = batch_size
        self.y_loss = y_loss
        self.s_loss = s_loss
        self.epochs = epochs
        self.adv_weight = adv_weight
        self.validation_pcnt = validation_pcnt

    def run(self, train: DataTuple, test: TestTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from ...implementations import beutel  # only import this on demand

        return beutel.train_and_transform(
            train,
            test,
            self.fairness,
            self.enc_size,
            self.adv_size,
            self.pred_size,
            self.enc_activation,
            self.adv_activation,
            self.batch_size,
            self.y_loss,
            self.s_loss,
            self.epochs,
            self.adv_weight,
            self.validation_pcnt,
        )

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_path: Path,
        new_test_path: Path,
    ) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_path, new_test_path, vars(self))
        return ["-m", "ethicml.implementations.beutel"] + args

    @property
    def name(self) -> str:
        return "Beutel"
