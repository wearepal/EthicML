"""Variational Fair Auto-Encoder by Louizos et al."""
from pathlib import Path
from typing import Dict, List, Optional, Union

from .interface import flag_interface
from .pre_algorithm import PreAlgorithmAsync

__all__ = ["VFAE"]


class VFAE(PreAlgorithmAsync):
    """VFAE Object - see implementation file for details."""

    def __init__(
        self,
        dataset: str,
        supervised: bool = True,
        epochs: int = 10,
        batch_size: int = 32,
        fairness: str = "DI",
        z1_enc_size: Optional[List[int]] = None,
        z2_enc_size: Optional[List[int]] = None,
        z1_dec_size: Optional[List[int]] = None,
    ):
        # pylint: disable=too-many-arguments
        super().__init__(name="VFAE")

        if z1_enc_size is None:
            z1_enc_size = [100]
        if z2_enc_size is None:
            z2_enc_size = [100]
        if z1_dec_size is None:
            z1_dec_size = [100]

        self.flags: Dict[str, Union[int, str, List[int]]] = {
            "supervised": supervised,
            "fairness": fairness,
            "batch_size": batch_size,
            "epochs": epochs,
            "dataset": dataset,
            "z1_enc_size": z1_enc_size,
            "z2_enc_size": z2_enc_size,
            "z1_dec_size": z1_dec_size,
        }

    def _script_command(
        self, train_path: Path, test_path: Path, new_train_path: Path, new_test_path: Path
    ) -> List[str]:
        args = flag_interface(train_path, test_path, new_train_path, new_test_path, self.flags)
        return ["-m", "ethicml.implementations.vfae"] + args
