"""Manually specified (i.e. not learned) models."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ranzen import implements

from ethicml.models.inprocess.in_algorithm import InAlgorithmNoParams
from ethicml.utility import DataTuple, Prediction, TestTuple

__all__ = ["Corels"]


@dataclass
class Corels(InAlgorithmNoParams):
    """CORELS (Certifiably Optimal RulE ListS) algorithm for the COMPAS dataset.

    This algorithm uses if-statements to make predictions. It only works on COMPAS with s as sex.

    From this paper: https://arxiv.org/abs/1704.01701
    """

    @property
    @implements(InAlgorithmNoParams)
    def name(self) -> str:
        return "CORELS"

    @implements(InAlgorithmNoParams)
    def fit(self, train: DataTuple, seed: int = 888) -> Corels:
        return self

    @implements(InAlgorithmNoParams)
    def predict(self, test: TestTuple) -> Prediction:
        if test.name is None or "Compas" not in test.name or test.s.name != "sex":
            raise RuntimeError("The Corels algorithm only works on the COMPAS dataset")
        age = test.x["age-num"].to_numpy()
        priors = test.x["priors-count"].to_numpy()
        sex = test.s.to_numpy()
        male = 1
        condition1 = (age >= 18) & (age <= 20) & (sex == male)
        condition2 = (age >= 21) & (age <= 23) & (priors >= 2) & (priors <= 3)
        condition3: np.ndarray = priors > 3
        pred = np.where(condition1 | condition2 | condition3, np.ones_like(age), np.zeros_like(age))
        return Prediction(hard=pd.Series(pred))

    @implements(InAlgorithmNoParams)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        return self.predict(test)
