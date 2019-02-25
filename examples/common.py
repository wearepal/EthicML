"""Functions that are common to the standalone algorithms"""
import sys
from pathlib import Path
from typing import List, Tuple, NamedTuple, Optional

import numpy as np
import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load dataframe from a parquet file"""
    with path.open('rb') as f:
        df = pd.read_parquet(f)
    return df


class DataTuple(NamedTuple):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels"""
    x: pd.DataFrame  # features
    s: pd.DataFrame  # senstitive attributes
    y: pd.DataFrame  # class labels


class CommonIn:
    """
    The common interface is a convention for how to pass the information about where the data is to
    the script. This class is for *in* algorithms.

    The convention is: x (train), s (train), y (train), x (test), s (test), predictions
    """
    def __init__(self, args: Optional[List[str]] = None):
        """
        Args:
            args: the commandline arguments WITHOUT the first element (which is usually the name of
                  the script)
        """
        self.args = sys.argv[1:] if args is None else args

    def load_data(self) -> Tuple[DataTuple, DataTuple]:
        """Load the data from the files"""
        train = DataTuple(
            x=load_dataframe(Path(self.args[0])),
            s=load_dataframe(Path(self.args[1])),
            y=load_dataframe(Path(self.args[2])),
        )
        test = DataTuple(
            x=load_dataframe(Path(self.args[3])),
            s=load_dataframe(Path(self.args[4])),
            y=pd.DataFrame(),  # an empty dataframe because the training labels are not given
        )
        return train, test

    def save_predictions(self, predictions: np.array):
        """Save the data to the file that was specified in the commandline arguments"""
        pred_path = Path(self.args[5])
        np.save(pred_path, predictions)
