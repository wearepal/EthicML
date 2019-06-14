"""
Variational Fair Auto-Encoder by Louizos et al
"""

from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import pandas as pd

from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithmAsync
from ethicml.algorithms.preprocess.interface import flag_interface
from ethicml.algorithms.utils import PathTuple, TestPathTuple, DataTuple, TestTuple


class VFAE(PreAlgorithmAsync):
    """
    VFAE Object - see implementation file for details
    """

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
        super().__init__()

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

    def run(self, train: DataTuple, test: TestTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from ...implementations.vfae import train_and_transform

        return train_and_transform(train, test, self.flags)

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_path: Path,
        new_test_path: Path,
    ) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_path, new_test_path, self.flags)
        return ["-m", "ethicml.implementations.vfae"] + args

    @property
    def name(self) -> str:
        return "VFAE"
