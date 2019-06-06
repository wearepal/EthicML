"""
Simply returns the majority label from the train set
"""

import pandas as pd

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm


class Majority(InAlgorithm):
    """Simply returns the majority label from the train set"""

    def run(self, train, test):
        maj = train.y.mode().values
        return pd.DataFrame(maj.repeat(len(test.x)), columns=["preds"])

    @property
    def name(self) -> str:
        return "Majority"
