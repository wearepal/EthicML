"""Simply returns the majority label from the train set."""
from dataclasses import dataclass

import pandas as pd
from ranzen import implements

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithmDC

__all__ = ["Majority"]


@dataclass
class Majority(InAlgorithmDC):
    """Simply returns the majority label from the train set."""

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "Majority"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple) -> InAlgorithmDC:
        self.maj = train.y.mode().iloc[0].to_numpy()  # type: ignore[attr-defined]
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.maj.repeat(len(test.x))))

    @implements(InAlgorithmDC)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        maj = train.y.mode().iloc[0].to_numpy()  # type: ignore[attr-defined]
        return Prediction(hard=pd.Series(maj.repeat(len(test.x))))
