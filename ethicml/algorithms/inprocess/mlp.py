"""Wrapper for SKLearn implementation of MLP."""
from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import numpy as np
import pandas as pd
from ranzen import implements
from sklearn.neural_network import MLPClassifier

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmDC
from ethicml.utility import DataTuple, Prediction, SoftPrediction, TestTuple

__all__ = ["MLP"]


@dataclass
class MLP(InAlgorithmDC):
    """Multi-layer Perceptron.

    This is a wraper around the SKLearn implementation of the MLP.
    Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

    :param hidden_layer_sizes: The number of neurons in each hidden layer.
    """

    is_fairness_algo: ClassVar[bool] = False
    hidden_layer_sizes: Tuple[int, ...] = field(
        default_factory=lambda: MLPClassifier().hidden_layer_sizes
    )

    @implements(InAlgorithmDC)
    def get_name(self) -> str:
        return "MLP"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple, seed: int = 888) -> InAlgorithmDC:
        self.clf = select_mlp(self.hidden_layer_sizes, seed=seed)
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        return SoftPrediction(soft=pd.Series(self.clf.predict_proba(test.x)[:, 1]))

    @implements(InAlgorithmDC)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        clf = select_mlp(self.hidden_layer_sizes, seed=seed)
        clf.fit(train.x, train.y.to_numpy().ravel())
        return SoftPrediction(soft=pd.Series(clf.predict_proba(test.x)[:, 1]))


def select_mlp(hidden_layer_sizes: Tuple[int, ...], seed: int) -> MLPClassifier:
    """Create MLP model for the given parameters.

    :param hidden_layer_sizes: The number of neurons in each hidden layer.
    :param seed: The seed for the random number generator.
    """
    random_state = np.random.RandomState(seed=seed)
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
