"""
Simply returns the majority label from the train set
"""

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.implementations import majority


class Majority(InAlgorithm):
    """Support Vector Machine"""
    def __init__(self):
        super().__init__()

    def _run(self, train, test):
        return majority.train_and_predict(train, test)

    def _script_command(self, train_paths, test_paths, pred_path):
        script = ['-m', majority.train_and_predict.__module__]
        args = conventional_interface(train_paths, test_paths, pred_path)
        return script + args

    @property
    def name(self) -> str:
        return "Majority"
