"""Simply returns the majority label from the train set."""

import pandas as pd
from kit import implements

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["Majority"]


class Majority(InAlgorithm):
    """Simply returns the majority label from the train set."""

    def __init__(self) -> None:
        super().__init__(name="Majority", is_fairness_algo=False)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        maj = train.y.mode().iloc[0].to_numpy()
        return Prediction(hard=pd.Series(maj.repeat(len(test.x))))
