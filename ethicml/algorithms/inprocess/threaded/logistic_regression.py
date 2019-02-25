"""Threaded logistic regression"""
import numpy as np
import pandas as pd

from .threaded_in_algorithm import ThreadedInAlgorithm


class ThreadedLR(ThreadedInAlgorithm):
    """Threaded logistic regression"""
    def run(self, train_paths, test_paths, tmp_path):
        # path where the predictions are supposed to be stored
        pred_path = tmp_path / "predictions.npy"

        # the following strings are passed to the other script
        args = [
            str(train_paths.x),
            str(train_paths.y),
            str(test_paths.x),
            str(pred_path)
        ]
        # call the script (this is blocking)
        self._call_script("examples/logistic_regression.py", args)
        # load the results
        predictions = np.load(pred_path)
        return pd.DataFrame(predictions, columns=["preds"])

    @property
    def name(self):
        return "threaded_LR"
