"""Simply returns a random (but legal) label."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ranzen import implements

from ethicml.algorithms.inprocess.in_algorithm import HyperParamType, InAlgorithm
from ethicml.utility import DataTuple, Prediction, TestTuple

__all__ = ["Blind"]


@dataclass
class Blind(InAlgorithm):
    """A Random classifier.

    This is useful as a baseline method and operates a 'coin flip' to assign a label.
    Returns a random label.
    """

    @implements(InAlgorithm)
    def get_name(self) -> str:
        return "Blind"

    @implements(InAlgorithm)
    def get_hyperparameters(self) -> HyperParamType:
        return {}

    @implements(InAlgorithm)
    def fit(self, train: DataTuple, seed: int = 888) -> InAlgorithm:
        self.vals = train.y.drop_duplicates()
        self.seed = seed
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        random = np.random.RandomState(self.seed)
        return Prediction(hard=pd.Series(random.choice(self.vals.to_numpy(), test.x.shape[0])))

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        vals = train.y.drop_duplicates()
        random = np.random.RandomState(seed)
        return Prediction(hard=pd.Series(random.choice(vals.to_numpy(), test.x.shape[0])))
