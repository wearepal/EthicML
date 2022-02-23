"""Simply returns a random (but legal) label."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ranzen import implements

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithmDC

__all__ = ["Blind"]


@dataclass
class Blind(InAlgorithmDC):
    """Returns a random label."""

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "Blind"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple) -> InAlgorithmDC:
        self.vals = train.y.drop_duplicates()
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        random = np.random.RandomState(self.seed)

        return Prediction(hard=pd.Series(random.choice(self.vals.T.to_numpy()[0], test.x.shape[0])))
