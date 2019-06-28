"""Beutel's algorithm"""
from pathlib import Path
from typing import List, Sequence, Tuple

from ethicml.utility.data_structures import DataTuple, TestTuple, PathTuple, TestPathTuple, FairType
from .pre_algorithm import PreAlgorithmAsync
from .interface import flag_interface


class Beutel(PreAlgorithmAsync):
    """Beutel's adversarially learned fair representations"""

    def __init__(
        self,
        fairness: FairType = FairType.DI,
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

    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        from ...implementations import beutel  # only import this on demand because of pytorch

        # SUGGESTION: it would be great if BeutelSettings could already be created in the init
        flags = beutel.BeutelSettings(
            fairness=self.fairness,
            enc_size=self.enc_size,
            adv_size=self.adv_size,
            pred_size=self.pred_size,
            enc_activation=self.enc_activation,
            adv_activation=self.adv_activation,
            batch_size=self.batch_size,
            y_loss=self.y_loss,
            s_loss=self.s_loss,
            epochs=self.epochs,
            adv_weight=self.adv_weight,
            validation_pcnt=self.validation_pcnt,
        )
        return beutel.train_and_transform(train, test, flags)

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_x_path: Path,
        new_train_s_path: Path,
        new_train_y_path: Path,
        new_train_name_path: Path,
        new_test_x_path: Path,
        new_test_s_path: Path,
        new_test_name_path: Path,
    ) -> List[str]:
        args = flag_interface(
            train_paths,
            test_paths,
            new_train_x_path,
            new_train_s_path,
            new_train_y_path,
            new_train_name_path,
            new_test_x_path,
            new_test_s_path,
            new_test_name_path,
            vars(self),
        )
        return ["-m", "ethicml.implementations.beutel"] + args

    @property
    def name(self) -> str:
        return "Beutel"
