"""How would a perfect predictor perform?"""
from dataclasses import dataclass

from ranzen import implements

from ethicml.utility import DataTuple, Prediction, TestTuple

from ..postprocess.dp_flip import DPFlip
from .in_algorithm import InAlgorithm, InAlgorithmDC

__all__ = ["Oracle", "DPOracle"]


@dataclass
class Oracle(InAlgorithmDC):
    """A perfect predictor.

    Can only be used if test is a DataTuple, rather than the usual TestTuple.
    This model isn't intended for general use,
    but can be useful if you want to either do a sanity check, or report potential values.
    """

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "Oracle"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple) -> InAlgorithm:
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        return Prediction(hard=test.y[test.y.columns[0]].copy())

    @implements(InAlgorithmDC)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        return Prediction(hard=test.y[test.y.columns[0]].copy())


class DPOracle(InAlgorithmDC):
    """A perfect Demographic Parity Predictor.

    Can only be used if test is a DataTuple, rather than the usual TestTuple.
    This model isn't intended for general use,
    but can be useful if you want to either do a sanity check, or report potential values.
    """

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "DemPar. Oracle"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple) -> InAlgorithm:
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        flipper = DPFlip(seed=self.seed)
        test_preds = Prediction(test.y[test.y.columns[0]].copy())
        return flipper.run(test_preds, test, test_preds, test)

    @implements(InAlgorithmDC)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        flipper = DPFlip(seed=self.seed)
        test_preds = Prediction(test.y[test.y.columns[0]].copy())
        return flipper.run(test_preds, test, test_preds, test)
