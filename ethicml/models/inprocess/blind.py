"""Simply returns a random (but legal) label."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from ranzen import implements

from ethicml.models.inprocess.in_algorithm import InAlgorithmNoParams
from ethicml.utility import DataTuple, Prediction, TestTuple

__all__ = ["Blind"]


@dataclass
class Blind(InAlgorithmNoParams):
    """A Random classifier.

    This is useful as a baseline method and operates a 'coin flip' to assign a label.
    Returns a random label.
    """

    is_fairness_algo: ClassVar[bool] = False

    @property
    @implements(InAlgorithmNoParams)
    def name(self) -> str:
        return "Blind"

    @implements(InAlgorithmNoParams)
    def fit(self, train: DataTuple, seed: int = 888) -> Blind:
        self.vals = train.y.drop_duplicates()
        self.seed = seed
        return self

    @implements(InAlgorithmNoParams)
    def predict(self, test: TestTuple) -> Prediction:
        random = np.random.RandomState(self.seed)
        return Prediction(hard=pd.Series(random.choice(self.vals.to_numpy(), test.x.shape[0])))

    @implements(InAlgorithmNoParams)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        vals = train.y.drop_duplicates()
        random = np.random.RandomState(seed)
        return Prediction(hard=pd.Series(random.choice(vals.to_numpy(), test.x.shape[0])))
