"""Useful functions used in implementations."""
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import tap

from ethicml.utility import DataTuple, PathTuple, TestPathTuple, TestTuple


class AlgoArgs(tap.Tap):
    """ArgumentParser that already has arguments for the input data paths.

    This is a quick way to create a parser that can parse the filenames for the data. This class
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


class PreAlgoArgs(AlgoArgs):
    """ArgumentParser for pre-processing algorithms."""

    # paths to where the processed inputs should be stored
    new_train_x: str
    new_train_s: str
    new_train_y: str
    new_train_name: str
    new_test_x: str
    new_test_s: str
    new_test_name: str


class InAlgoArgs(AlgoArgs):
    """ArgumentParser that already has arguments for the paths needed for InAlgorithms."""

    # path to where the predictions should be stored
    pred_path: str


def save_predictions(predictions: Union[np.ndarray, pd.DataFrame], args: InAlgoArgs):
    """Save the data to the file that was specified in the commandline arguments."""
    if not isinstance(predictions, pd.DataFrame):
        df = pd.DataFrame(predictions, columns=["pred"])
    else:
        df = predictions
    pred_path = Path(args.pred_path)
    df.to_feather(pred_path)


def load_data_from_flags(flags: AlgoArgs) -> Tuple[DataTuple, TestTuple]:
    """Load data from the paths specified in the flags."""
    train_paths = PathTuple(
        x=Path(flags.train_x), s=Path(flags.train_s), y=Path(flags.train_y), name=flags.train_name
    )
    test_paths = TestPathTuple(x=Path(flags.test_x), s=Path(flags.test_s), name=flags.test_name)
    return train_paths.load_from_feather(), test_paths.load_from_feather()


def save_transformations(transforms: Tuple[DataTuple, TestTuple], args: PreAlgoArgs):
    """Save the data to the file that was specified in the commandline arguments."""
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
