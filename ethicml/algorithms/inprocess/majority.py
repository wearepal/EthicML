"""
Simply returns the majority label from the train set
"""

import pandas as pd

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.utility.data_structures import Predictions


class Majority(InAlgorithm):
    """Simply returns the majority label from the train set"""

    def run(self, train, test):
        maj = train.y.mode().values
        labels = pd.DataFrame(maj.repeat(len(test.x)), columns=["preds"])
        return Predictions(soft=labels, hard=labels)

    @property
    def name(self) -> str:
        return "Majority"
