"""Functions that are common to the threaded algorithms"""
import sys
from pathlib import Path
from typing import Tuple, Union, List, Optional

import pandas as pd
import numpy as np

from ..algorithms.utils import DataTuple
from ..algorithms.algorithm_base import load_dataframe


class InAlgoInterface:
    """Commandline "interface" for in-process algorithms"""
    def __init__(self, args: Optional[List[str]] = None):
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
            y=load_dataframe(Path(self.args[5])),
        )
        return train, test

    def save_predictions(self, predictions: Union[np.ndarray, pd.DataFrame]):
        """Save the data to the file that was specified in the commandline arguments"""
        if not isinstance(predictions, pd.DataFrame):
            df = pd.DataFrame(predictions, columns=["pred"])
        else:
            df = predictions
        pred_path = Path(self.args[6])
        df.to_feather(pred_path)

    def remaining_args(self):
        """Additional commandline arguments beyond the data paths and the prediction path"""
        return self.args[7:]


def load_data_from_flags(flags):
    """Load data from the paths specified in the flags"""
    train = DataTuple(
        x=load_dataframe(Path(flags['train_x'])),
        s=load_dataframe(Path(flags['train_s'])),
        y=load_dataframe(Path(flags['train_y'])),
    )
    test = DataTuple(
        x=load_dataframe(Path(flags['test_x'])),
        s=load_dataframe(Path(flags['test_s'])),
        y=load_dataframe(Path(flags['test_y'])),
    )
    return train, test


def save_transformations(transforms: Tuple[pd.DataFrame, pd.DataFrame],
                         transform_paths: Tuple[str, str]):
    """Save the data to the file that was specified in the commandline arguments"""
    assert isinstance(transforms[0], pd.DataFrame)
    assert isinstance(transforms[1], pd.DataFrame)
    for transform, path in zip(transforms, transform_paths):
        transform_path = Path(path)
        # convert the column IDs to strings because the feather format requires that
        transform.columns = transform.columns.astype(str)
        transform.to_feather(transform_path)
