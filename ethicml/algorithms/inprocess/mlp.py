"""Wrapper for SKLearn implementation of MLP."""
from pathlib import Path
from typing import Optional, Tuple, List

from sklearn.neural_network import MLPClassifier
import pandas as pd

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.implementations import mlp
from ethicml.utility.data_structures import (
    ActivationType,
    DataTuple,
    TestTuple,
    PathTuple,
    TestPathTuple,
)
from .shared import conventional_interface


class MLP(InAlgorithmAsync):
    """Multi-layer Perceptron."""

    def __init__(
        self,
        hidden_layer_sizes: Optional[Tuple[int]] = None,
        activation: Optional[ActivationType] = None,
    ):
        super().__init__()
        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = MLPClassifier().hidden_layer_sizes
        else:
            self.hidden_layer_sizes = hidden_layer_sizes
        self.activation: ActivationType = (
            MLPClassifier().activation if activation is None else activation
        )

    def run(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        return mlp.train_and_predict(train, test, self.hidden_layer_sizes, self.activation)

    def _script_command(
        self, train_paths: PathTuple, test_paths: TestPathTuple, pred_path: Path
    ) -> List[str]:
        script = ["-m", mlp.train_and_predict.__module__]
        args = conventional_interface(
            train_paths, test_paths, pred_path, str(self.hidden_layer_sizes), str(self.activation)
        )
        return script + args

    @property
    def name(self) -> str:
        return "MLP"
