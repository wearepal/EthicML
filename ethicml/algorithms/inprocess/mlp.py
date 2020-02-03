"""Wrapper for SKLearn implementation of MLP."""
from typing import Optional, Tuple, Dict

from sklearn.neural_network import MLPClassifier
import pandas as pd

from ethicml.common import implements
from ethicml.utility.data_structures import ActivationType, DataTuple, TestTuple, Prediction
from .in_algorithm import InAlgorithm


ACTIVATIONS: Dict[str, ActivationType] = {
    "identity": "identity",
    "logistic": "logistic",
    "tanh": "tanh",
    "relu": "relu",
}


class MLP(InAlgorithm):
    """Multi-layer Perceptron."""

    def __init__(
        self,
        hidden_layer_sizes: Optional[Tuple[int]] = None,
        activation: Optional[ActivationType] = None,
    ):
        """Init MLP."""
        super().__init__(is_fairness_algo=False)
        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = MLPClassifier().hidden_layer_sizes
        else:
            self.hidden_layer_sizes = hidden_layer_sizes
        self.activation: ActivationType = (
            MLPClassifier().activation if activation is None else activation
        )

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        clf = select_mlp(self.hidden_layer_sizes, self.activation)
        clf.fit(train.x, train.y.to_numpy().ravel())
        return Prediction(hard=pd.Series(clf.predict(test.x)))

    @property
    def name(self) -> str:
        """Getter for algorithm name."""
        return "MLP"


def select_mlp(hidden_layer_sizes: Tuple[int], activation: ActivationType) -> MLPClassifier:
    """Create MLP model for the given parameters."""
    assert activation in ACTIVATIONS

    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=888
    )
