"""Threaded logistic regression"""
import sys
from subprocess import call
from typing import Optional

import numpy as np
import pandas as pd

from .threaded_in_algorithm import ThreadedInAlgorithm


class ThreadedLR(ThreadedInAlgorithm):
    """Threaded logistic regression"""
    def __init__(self, python_exe: Optional[str]=None):
        if python_exe is None:
            # use the python executable that this script was called with
            python_exe = sys.executable
        self.python_exe = python_exe

    def run(self, train_paths, test_paths, tmp_path):
        # path where the predictions are supposed to be stored
        pred_path = tmp_path / "predictions.npy"

        args = [
            self.python_exe,
            "examples/logistic_regression.py",
            # the following strings are passed to the other script
            str(train_paths.x),
            str(train_paths.y),
            str(test_paths.x),
            str(pred_path)
        ]
        # call the script (this is blocking)
        call(args)
        # load the results
        predictions = np.load(pred_path)
        return pd.DataFrame(predictions, columns=["preds"])

    @property
    def name(self):
        return "threaded_LR"
