"""Simply returns the majority label from the train set."""

import pandas as pd
from ranzen import implements

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["Majority"]


class Majority(InAlgorithm):
    """Simply returns the majority label from the train set."""

    def __init__(self, seed: int = 888) -> None:
        super().__init__(name="Majority", is_fairness_algo=False, seed=seed)

    @implements(InAlgorithm)
    def fit(self, train: DataTuple) -> InAlgorithm:
        self.maj = train.y.mode().iloc[0].to_numpy()  # type: ignore[attr-defined]
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.maj.repeat(len(test.x))))

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        maj = train.y.mode().iloc[0].to_numpy()  # type: ignore[attr-defined]
        return Prediction(hard=pd.Series(maj.repeat(len(test.x))))
