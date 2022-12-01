"""Simply returns the majority label from the train set."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

import pandas as pd

from ethicml.models.inprocess.in_algorithm import InAlgorithmNoParams
from ethicml.utility import DataTuple, Prediction, TestTuple

__all__ = ["Majority"]


@dataclass
class Majority(InAlgorithmNoParams):
    """Simply returns the majority label from the train set."""

    is_fairness_algo: ClassVar[bool] = False

    @property
    @override
    def name(self) -> str:
        return "Majority"

    @override
    def fit(self, train: DataTuple, seed: int = 888) -> Majority:
        self.maj = train.y.mode(dropna=True).to_numpy()
        return self

    @override
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.maj.repeat(len(test.x))))

    @override
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        maj = train.y.mode(dropna=True).to_numpy()
        return Prediction(hard=pd.Series(maj.repeat(len(test.x))))
