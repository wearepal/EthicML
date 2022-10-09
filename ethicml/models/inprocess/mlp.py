"""Wrapper for SKLearn implementation of MLP."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import numpy as np
from ranzen import implements
from sklearn.neural_network import MLPClassifier

from ethicml.models.inprocess.in_algorithm import InAlgorithmDC
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
    batch_size: int = 32
    lr: float = 1e-3

    @property
    @implements(InAlgorithmDC)
    def name(self) -> str:
        return "MLP"

    @implements(InAlgorithmDC)
    def fit(self, train: DataTuple, seed: int = 888) -> MLP:
        self.clf = select_mlp(
            self.hidden_layer_sizes, seed=seed, lr=self.lr, batch_size=self.batch_size
        )
        self.clf.fit(train.x, train.y.to_numpy().ravel())
        return self

    @implements(InAlgorithmDC)
    def predict(self, test: TestTuple) -> Prediction:
        return SoftPrediction(soft=self.clf.predict_proba(test.x), info=self.hyperparameters)

    @implements(InAlgorithmDC)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        clf = select_mlp(self.hidden_layer_sizes, seed=seed, lr=self.lr, batch_size=self.batch_size)
        clf.fit(train.x, train.y.to_numpy().ravel())
        return SoftPrediction(soft=clf.predict_proba(test.x), info=self.hyperparameters)


def select_mlp(
    hidden_layer_sizes: tuple[int, ...], seed: int, lr: float, batch_size: int
) -> MLPClassifier:
    """Create MLP model for the given parameters.

    :param hidden_layer_sizes: The number of neurons in each hidden layer.
    :param seed: The seed for the random number generator.
    :param lr: The learning rate.
    :param batch_size: The batch size.
    :returns: The instantiated MLP.
    """
    random_state = np.random.RandomState(seed=seed)
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        solver='adam',
        random_state=random_state,
        learning_rate_init=lr,
        batch_size=batch_size,
    )
