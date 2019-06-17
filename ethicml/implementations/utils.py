"""
Useful functions used in implementations
"""
import sys
import argparse
from pathlib import Path
from typing import Tuple, Union, List, Optional, Dict, Any

import pandas as pd
import numpy as np

from ethicml.algorithms.utils import DataTuple, TestTuple
from ethicml.algorithms.algorithm_base import load_dataframe


class InAlgoInterface:
    """Commandline "interface" for in-process algorithms"""

    def __init__(self, args: Optional[List[str]] = None):
        self.args: List[str] = sys.argv[1:] if args is None else args

    def load_data(self) -> Tuple[DataTuple, TestTuple]:
        """Load the data from the files"""
        train = DataTuple(
            x=load_dataframe(Path(self.args[0])),
            s=load_dataframe(Path(self.args[1])),
            y=load_dataframe(Path(self.args[2])),
        )
        test = TestTuple(x=load_dataframe(Path(self.args[3])), s=load_dataframe(Path(self.args[4])))
        return train, test

    def save_predictions(self, predictions: Union[np.ndarray, pd.DataFrame]):
        """Save the data to the file that was specified in the commandline arguments"""
        if not isinstance(predictions, pd.DataFrame):
            df = pd.DataFrame(predictions, columns=["pred"])
        else:
            df = predictions
        pred_path = Path(self.args[5])
        df.to_feather(pred_path)

    def remaining_args(self) -> List[str]:
        """Additional commandline arguments beyond the data paths and the prediction path"""
        return self.args[6:]


def load_data_from_flags(flags: Dict[str, Any]) -> Tuple[DataTuple, TestTuple]:
    """Load data from the paths specified in the flags"""
    train = DataTuple(
        x=load_dataframe(Path(flags["train_x"])),
        s=load_dataframe(Path(flags["train_s"])),
        y=load_dataframe(Path(flags["train_y"])),
    )
    test = TestTuple(
        x=load_dataframe(Path(flags['test_x'])), s=load_dataframe(Path(flags['test_s']))
    )
    return train, test


def save_transformations(
    transforms: Tuple[pd.DataFrame, pd.DataFrame], transform_paths: Tuple[str, str]
):
    """Save the data to the file that was specified in the commandline arguments"""
    assert isinstance(transforms[0], pd.DataFrame)
    assert isinstance(transforms[1], pd.DataFrame)
    for transform, path in zip(transforms, transform_paths):
        transform_path = Path(path)
        # convert the column IDs to strings because the feather format requires that
        transform.columns = transform.columns.astype(str)
        transform.to_feather(transform_path)


def pre_algo_argparser() -> argparse.ArgumentParser:
    """ArgumentParser that already has arguments for the paths

    This is a quick way to create a parser that can parse the filenames for the data. This function
    can be used by pre-algorithms to implement a commandline interface.
    """
    parser = argparse.ArgumentParser()

    # paths to the files with the data
    parser.add_argument("--train_x", required=True)
    parser.add_argument("--train_s", required=True)
    parser.add_argument("--train_y", required=True)
    parser.add_argument("--test_x", required=True)
    parser.add_argument("--test_s", required=True)

    # paths to where the processed inputs should be stored
    parser.add_argument("--train_new", required=True)
    parser.add_argument("--test_new", required=True)
    return parser
