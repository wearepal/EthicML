"""Perfect predictors."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.models.inprocess.in_algorithm import InAlgorithmNoParams
from ethicml.utility import DataTuple, Prediction, TestTuple

from ..postprocess.dp_flip import DPFlip

__all__ = ["Oracle", "DPOracle"]


@dataclass
class Oracle(InAlgorithmNoParams):
    """A perfect predictor.

    Can only be used if test is a DataTuple, rather than the usual TestTuple.
    This model isn't intended for general use,
    but can be useful if you want to either do a sanity check, or report potential values.
    """

    is_fairness_algo: ClassVar[bool] = False

    @property
    @implements(InAlgorithmNoParams)
    def name(self) -> str:
        return "Oracle"

    @implements(InAlgorithmNoParams)
    def fit(self, train: DataTuple, seed: int = 888) -> Oracle:
        return self

    @implements(InAlgorithmNoParams)
    def predict(self, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        return Prediction(hard=test.y.copy())

    @implements(InAlgorithmNoParams)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        return Prediction(hard=test.y.copy())


@dataclass
class DPOracle(InAlgorithmNoParams):
    """A perfect Demographic Parity Predictor.

    Can only be used if test is a DataTuple, rather than the usual TestTuple.
    This model isn't intended for general use,
    but can be useful if you want to either do a sanity check, or report potential values.
    """

    @property
    @implements(InAlgorithmNoParams)
    def name(self) -> str:
        return "DemPar. Oracle"

    @implements(InAlgorithmNoParams)
    def fit(self, train: DataTuple, seed: int = 888) -> DPOracle:
        self.seed = seed
        return self

    @implements(InAlgorithmNoParams)
    def predict(self, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        flipper = DPFlip()
        test_preds = Prediction(test.y.copy())
        return flipper.run(test_preds, test, test_preds, test, seed=self.seed)

    @implements(InAlgorithmNoParams)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        flipper = DPFlip()
        test_preds = Prediction(test.y.copy())
        return flipper.run(test_preds, test, test_preds, test, seed=seed)
