"""Simply returns the majority label from the train set."""

import pandas as pd

from ethicml.common import implements
from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.utility.data_structures import DataTuple, TestTuple, Prediction


class Majority(InAlgorithm):
    """Simply returns the majority label from the train set."""

    def __init__(self) -> None:
        """Init Majority."""
        super().__init__(is_fairness_algo=False)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        maj = train.y.mode().to_numpy()
        return Prediction(hard=pd.Series(maj.repeat(len(test.x))))

    @property
    def name(self) -> str:
        """Getter for algorithm name."""
        return "Majority"
