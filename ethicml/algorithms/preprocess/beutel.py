"""Beutel's algorithm."""
from typing import List, Sequence, Union, Dict

from ethicml.utility.data_structures import PathTuple, TestPathTuple, FairnessType
from .pre_algorithm import PreAlgorithmAsync
from .interface import flag_interface


class Beutel(PreAlgorithmAsync):
    """Beutel's adversarially learned fair representations."""

    def __init__(
        self,
        fairness: FairnessType = "DP",
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
        """Init Beutel."""
        # pylint: disable=too-many-arguments
        super().__init__()
        self.flags: Dict[str, Union[str, Sequence[int], int, float]] = {
            "fairness": fairness,
            "enc_size": enc_size,
            "adv_size": adv_size,
            "pred_size": pred_size,
            "enc_activation": enc_activation,
            "adv_activation": adv_activation,
            "batch_size": batch_size,
            "y_loss": y_loss,
            "s_loss": s_loss,
            "epochs": epochs,
            "adv_weight": adv_weight,
            "validation_pcnt": validation_pcnt,
        }

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_paths: PathTuple,
        new_test_paths: TestPathTuple,
    ) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_paths, new_test_paths, self.flags)
        return ["-m", "ethicml.implementations.beutel"] + args

    @property
    def name(self) -> str:
        """Getter for algorithm name."""
        return "Beutel"
