"""Useful functions used in implementations."""
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import numpy as np
import tap

from ethicml.utility import DataTuple, TestTuple, PathTuple, TestPathTuple, save_helper


class AlgoArgs(tap.Tap):
    """ArgumentParser that already has arguments for the input data paths.

    This is a quick way to create a parser that can parse the filenames for the data. This class
    can be used by pre-algorithms to implement a commandline interface.
    """

    # paths to the files with the data
    train_data_path: str
    train_name: str
    test_data_path: str
    test_name: str


class PreAlgoArgs(AlgoArgs):
    """ArgumentParser for pre-processing algorithms."""

    # paths to where the processed inputs should be stored
    new_train: str
    new_train_name: str
    new_test: str
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
    np.savez(pred_path, pred=df.to_numpy())


def load_data_from_flags(flags: AlgoArgs) -> Tuple[DataTuple, TestTuple]:
    """Load data from the paths specified in the flags."""
    train_paths = PathTuple(data_path=Path(flags.train_data_path), name=flags.train_name)
    test_paths = TestPathTuple(data_path=Path(flags.test_data_path), name=flags.test_name)
    return train_paths.load_from_feather(), test_paths.load_from_feather()


def save_transformations(transforms: Tuple[DataTuple, TestTuple], args: PreAlgoArgs):
    """Save the data to the file that was specified in the commandline arguments."""
    assert isinstance(transforms[0], DataTuple)
    assert isinstance(transforms[1], TestTuple)

    train, test = transforms
    save_helper(Path(args.new_train), dict(x=train.x, s=train.s, y=train.y))
    save_helper(Path(args.new_test), dict(x=test.x, s=test.s))
