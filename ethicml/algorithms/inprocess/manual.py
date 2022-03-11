"""Manually specified (i.e. not learned) models."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ranzen import implements

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithmDC

__all__ = ["Corels"]


@dataclass
class Corels(InAlgorithmDC):
    """CORELS (Certifiably Optimal RulE ListS) algorithm for the COMPAS dataset.

    This algorithm uses if-statements to make predictions. It only works on COMPAS with s as sex.

    From this paper: https://arxiv.org/abs/1704.01701
    """

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "CORELS"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple) -> InAlgorithmDC:
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        if test.name is None or "Compas" not in test.name or "sex" not in test.s.columns:
            raise RuntimeError("The Corels algorithm only works on the COMPAS dataset")
        age = test.x["age-num"].to_numpy()
        priors = test.x["priors-count"].to_numpy()
        sex = test.s["sex"].to_numpy()
        male = 1
        condition1 = (age >= 18) & (age <= 20) & (sex == male)
        condition2 = (age >= 21) & (age <= 23) & (priors >= 2) & (priors <= 3)
        condition3: np.ndarray = priors > 3
        pred = np.where(condition1 | condition2 | condition3, np.ones_like(age), np.zeros_like(age))
        return Prediction(hard=pd.Series(pred))
