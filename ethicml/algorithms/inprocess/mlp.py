"""Wrapper for SKLearn implementation of MLP."""
from typing import ClassVar, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ranzen import implements
from sklearn.neural_network import MLPClassifier

from ethicml.utility import ActivationType, DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm

__all__ = ["MLP"]


ACTIVATIONS: Dict[str, ActivationType] = {
    "identity": ActivationType.identity,
    "logistic": ActivationType.logistic,
    "tanh": ActivationType.tanh,
    "relu": ActivationType.relu,
}


class MLP(InAlgorithm):
    """Multi-layer Perceptron.

    This is a wraper around the SKLearn implementation of the MLP.
    Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    """

    is_fairness_algo: ClassVar[bool] = False

    def __init__(
        self,
        *,
        hidden_layer_sizes: Optional[Tuple[int, ...]] = None,
        activation: Optional[ActivationType] = None,
        seed: int = 888,
    ):
        """Multi-Layer Perceptron.

        Args:
            hidden_layer_sizes: The number of neurons in each hidden layer.
            activation: The activation function to use.
            seed: The seed for the random number generator.
        """
        self.seed = seed
        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = MLPClassifier().hidden_layer_sizes
        else:
            self.hidden_layer_sizes = hidden_layer_sizes
        self.activation: ActivationType = (
            MLPClassifier().activation if activation is None else activation
        )
        self._hyperparameters = {
            "hidden_layer_sizes": f"{self.hidden_layer_sizes}",
            "activation": self.activation,
        }

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "MLP"

    @implements(InAlgorithm)
    def fit(self, train: DataTuple) -> InAlgorithm:
        self.clf = select_mlp(self.hidden_layer_sizes, self.activation, seed=self.seed)
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return Prediction(hard=pd.Series(self.clf.predict(test.x)))


def select_mlp(
    hidden_layer_sizes: Tuple[int, ...], activation: ActivationType, seed: int
) -> MLPClassifier:
    """Create MLP model for the given parameters."""
    assert activation in ACTIVATIONS.values()

    random_state = np.random.RandomState(seed=seed)
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=random_state
    )
