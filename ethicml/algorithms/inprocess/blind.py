"""Simply returns a random (but legal) label."""

import numpy as np
import pandas as pd
from kit import implements

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["Blind"]


class Blind(InAlgorithm):
    """Returns a random label."""

    def __init__(self, seed: int = 888) -> None:
        super().__init__(name="Blind", is_fairness_algo=False)
        self.seed = seed

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        train_y_vals = train.y.drop_duplicates()

        random = np.random.RandomState(self.seed)

        return Prediction(
            hard=pd.Series(random.choice(train_y_vals.T.to_numpy()[0], test.x.shape[0]))
        )
