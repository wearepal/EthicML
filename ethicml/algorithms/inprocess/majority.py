"""Simply returns the majority label from the train set."""

import pandas as pd

from ethicml.common import implements
from ethicml.utility.data_structures import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm


class Majority(InAlgorithm):
    """Simply returns the majority label from the train set."""

    def __init__(self) -> None:
        """Init Majority."""
        super().__init__(name="Majority", is_fairness_algo=False)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        maj = train.y.mode().iloc[0].to_numpy()
        return Prediction(hard=pd.Series(maj.repeat(len(test.x))))
