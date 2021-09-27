"""How would a perfect predictor perform?"""
from kit import implements, parsable

from ethicml.algorithms.postprocess.dp_flip import DPFlip
from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["Oracle", "DPOracle"]


class Oracle(InAlgorithm):
    """A perfect predictor.

    Can only be used if test is a DataTuple, rather than the usual TestTuple.
    This model isn't intended for general use,
    but can be useful if you want to either do a sanity check, or report potential values.
    """

    @parsable
    @implements(InAlgorithm)
    def __init__(self) -> None:
        super().__init__(name="Oracle", is_fairness_algo=False)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        return Prediction(hard=test.y[test.y.columns[0]].copy())


class DPOracle(InAlgorithm):
    """A perfect Demographic Parity Predictor.

    Can only be used if test is a DataTuple, rather than the usual TestTuple.
    This model isn't intended for general use,
    but can be useful if you want to either do a sanity check, or report potential values.
    """

    @parsable
    @implements(InAlgorithm)
    def __init__(self) -> None:
        super().__init__(name="DemPar. Oracle", is_fairness_algo=True)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        flipper = DPFlip()
        test_preds = Prediction(test.y[test.y.columns[0]].copy())
        return flipper.run(test_preds, test, test_preds, test)
