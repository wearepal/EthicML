"""Beutel's algorithm"""
from typing import Optional, List, Dict, Union

from .pre_algorithm import PreAlgorithm
from .interface import flag_interface


class Beutel(PreAlgorithm):
    """Beutel's adversarially learned fair representations"""
    def __init__(self,
                 fairness: str = "DI",
                 enc_size: Optional[List[int]] = None,
                 adv_size: Optional[List[int]] = None,
                 pred_size: Optional[List[int]] = None,
                 enc_activation: str = "Sigmoid()",
                 adv_activation: str = "Sigmoid()",
                 batch_size: int = 64,
                 y_loss: str = "BCELoss()",
                 s_loss: str = "BCELoss()",
                 epochs: int = 50):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.flags: Dict[str, Union[int, str, List[int]]] = {
            'fairness': fairness,
            'enc_size': [40] if enc_size is None else enc_size,
            'adv_size': [40] if adv_size is None else adv_size,
            'pred_size': [40] if pred_size is None else pred_size,
            'enc_activation': enc_activation,
            'adv_activation': adv_activation,
            'batch_size': batch_size,
            'y_loss': y_loss,
            's_loss': s_loss,
            'epochs': epochs,
        }

    def _run(self, train, test):
        from ...implementations import beutel  # only import this on demand
        return beutel.train_and_transform(train, test, self.flags)

    def _script_command(self, train_paths, test_paths, new_train_path, new_test_path):
        args = flag_interface(train_paths, test_paths, new_train_path, new_test_path, self.flags)
        return ['-m', 'ethicml.implementations.beutel'] + args

    @property
    def name(self) -> str:
        return "Beutel"
