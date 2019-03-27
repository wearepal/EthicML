"""
Implementation of cross validated LR. This is a work around for now,
long term we'll have a proper cross-validation mechanism
"""
from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.implementations import logistic_regression_cross_validated


class LRCV(InAlgorithm):
    """Kind of a cheap hack for now, but gives a proper cross-valudeted LR"""
    def _run(self, train, test):
        return logistic_regression_cross_validated.train_and_predict(train, test)

    def _script_command(self, train_paths, test_paths, pred_path):
        script = ['-m', logistic_regression_cross_validated.train_and_predict.__module__]
        args = conventional_interface(train_paths, test_paths, pred_path)
        return script + args

    @property
    def name(self) -> str:
        return "LRCV"
