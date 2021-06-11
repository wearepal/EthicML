"""How would a perfect predictor perform?"""
from kit import implements

from ethicml import DataTuple, InAlgorithm, Prediction, TestTuple
from ethicml.algorithms.postprocess.dp_flip import DPFlip


class Oracle(InAlgorithm):
    """A perfect predictor.

    Can only be used if test is a DataTuple, rather than the usual TestTuple.
    This model isn't intended for general use,
    but can be useful if you want to either do a sanity check, or report potential values.
    """

    @implements(InAlgorithm)
    def __init__(self) -> None:
        super().__init__(name="Oracle", is_fairness_algo=False)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        return Prediction(hard=test.y[test.y.columns[0]])


class DPOracle(InAlgorithm):
    """A perfect Demographic Parity Predictor.

    Can only be used if test is a DataTuple, rather than the usual TestTuple.
    This model isn't intended for general use,
    but can be useful if you want to either do a sanity check, or report potential values.
    """

    @implements(InAlgorithm)
    def __init__(self) -> None:
        super().__init__(name="DemPar. Oracle", is_fairness_algo=True)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        assert isinstance(test, DataTuple), "test must be a DataTuple."
        flipper = DPFlip()
        test_preds = Prediction(test.y[test.y.columns[0]])
        return flipper.run(test_preds, test, test_preds, test)
