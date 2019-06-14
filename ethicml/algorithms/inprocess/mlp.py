"""
Wrapper for SKLearn implementation of MLP
"""
from typing import Optional, Tuple

from sklearn.neural_network import MLPClassifier

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.implementations import mlp


class MLP(InAlgorithmAsync):
    """MLP"""

    def __init__(
        self, hidden_layer_sizes: Optional[Tuple[int]] = None, activation: Optional[str] = None
    ):
        super().__init__()
        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = MLPClassifier().hidden_layer_sizes
        else:
            self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = MLPClassifier().activation if activation is None else activation

    def run(self, train, test):
        return mlp.train_and_predict(train, test, self.hidden_layer_sizes, self.activation)

    def _script_command(self, train_paths, test_paths, pred_path):
        script = ["-m", mlp.train_and_predict.__module__]
        args = conventional_interface(
            train_paths, test_paths, pred_path, str(self.hidden_layer_sizes), str(self.activation)
        )
        return script + args

    @property
    def name(self) -> str:
        return "MLP"
