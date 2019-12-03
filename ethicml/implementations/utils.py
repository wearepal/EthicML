"""
Useful functions used in implementations
"""
import sys
from pathlib import Path
from typing import Tuple, Union, List, Optional, Dict, Any

import pandas as pd
import numpy as np
import tap

from ethicml.utility.data_structures import DataTuple, TestTuple, PathTuple, TestPathTuple


class PreAlgoArgs(tap.Tap):
    """ArgumentParser that already has arguments for the paths

    This is a quick way to create a parser that can parse the filenames for the data. This function
    can be used by pre-algorithms to implement a commandline interface.
    """

    # paths to the files with the data
    train_x: str
    train_s: str
    train_y: str
    train_name: str
    test_x: str
    test_s: str
    test_name: str

    # paths to where the processed inputs should be stored
    new_train_x: str
    new_train_s: str
    new_train_y: str
    new_train_name: str
    new_test_x: str
    new_test_s: str
    new_test_name: str


class InAlgoInterface:
    """Commandline "interface" for in-process algorithms"""

    def __init__(self, args: Optional[List[str]] = None):
        self.args: List[str] = sys.argv[1:] if args is None else args

    def load_data(self) -> Tuple[DataTuple, TestTuple]:
        """Load the data from the files"""
        train_paths = PathTuple(
            x=Path(self.args[0]), s=Path(self.args[1]), y=Path(self.args[2]), name=self.args[3]
        )
        test_paths = TestPathTuple(x=Path(self.args[4]), s=Path(self.args[5]), name=self.args[6])
        return train_paths.load_from_feather(), test_paths.load_from_feather()

    def save_predictions(self, predictions: Union[np.ndarray, pd.DataFrame]):
        """Save the data to the file that was specified in the commandline arguments"""
        if not isinstance(predictions, pd.DataFrame):
            df = pd.DataFrame(predictions, columns=["pred"])
        else:
            df = predictions
        pred_path = Path(self.args[7])
        df.to_feather(pred_path)

    def remaining_args(self) -> List[str]:
        """Additional commandline arguments beyond the data paths and the prediction path"""
        return self.args[8:]


def load_data_from_flags(flags: Dict[str, Any]) -> Tuple[DataTuple, TestTuple]:
    """Load data from the paths specified in the flags"""
    train_paths = PathTuple(
        x=Path(flags["train_x"]),
        s=Path(flags["train_s"]),
        y=Path(flags["train_y"]),
        name=flags["train_name"],
    )
    test_paths = TestPathTuple(
        x=Path(flags['test_x']), s=Path(flags['test_s']), name=flags["test_name"]
    )
    return train_paths.load_from_feather(), test_paths.load_from_feather()


def save_transformations(transforms: Tuple[DataTuple, TestTuple], args: PreAlgoArgs):
    """Save the data to the file that was specified in the commandline arguments"""
    assert isinstance(transforms[0], DataTuple)
    assert isinstance(transforms[1], TestTuple)

    train, test = transforms

    def _save(data: pd.DataFrame, path: str) -> None:
        # write the file
        data.columns = data.columns.astype(str)
        data.to_feather(Path(path))

    _save(train.x, args.new_train_x)
    _save(train.s, args.new_train_s)
    _save(train.y, args.new_train_y)
    _save(test.x, args.new_test_x)
    _save(test.s, args.new_test_s)
