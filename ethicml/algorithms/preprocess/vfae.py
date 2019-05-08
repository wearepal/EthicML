"""
Variational Fair Auto-Encoder by Louizos et al
"""

from pathlib import Path
from typing import List, Tuple, Dict, Union

import pandas as pd

from ethicml.algorithms.preprocess import PreAlgorithm
from ethicml.algorithms.preprocess.interface import flag_interface
from ethicml.algorithms.utils import PathTuple, DataTuple


class VFAE(PreAlgorithm):
    """
    VFAE Object - see implementation file for details
    """
    def __init__(self, epochs: int, batch_size: int, fairness: str, dataset: str):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.flags: Dict[str, Union[int, str, List[int]]] = {
            'fairness': fairness,
            'batch_size': batch_size,
            'epochs': epochs,
            'dataset': dataset
        }

    def _run(self, train: DataTuple, test: DataTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from ...implementations.vfae import train_and_transform
        return train_and_transform(train, test, self.flags)

    def _script_command(self, train_paths: PathTuple, test_paths: PathTuple, new_train_path: Path,
                        new_test_path: Path) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_path, new_test_path, self.flags)
        return ['-m', 'ethicml.implementations.vfae'] + args

    @property
    def name(self) -> str:
        return "VFAE"
