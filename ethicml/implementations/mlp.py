"""Implementation of MLP (actually just a wrapper around sklearn)"""
from typing import Tuple, Dict
from ast import literal_eval as make_tuple

from sklearn.neural_network import MLPClassifier

import pandas as pd

from ethicml.utility import ActivationType, DataTuple, TestTuple
from ethicml.implementations.utils import InAlgoInterface

ACTIVATIONS: Dict[str, ActivationType] = {
    "identity": "identity",
    "logistic": "logistic",
    "tanh": "tanh",
    "relu": "relu",
}


def select_mlp(hidden_layer_sizes: Tuple[int], activation: ActivationType):
    """Create MLP model for the given parameters"""

    assert activation in ACTIVATIONS

    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=888
    )


def train_and_predict(train: DataTuple, test: TestTuple, hid_layers, activation: ActivationType):
    """Train an SVM model and compute predictions on the given test data"""
    clf = select_mlp(hid_layers, activation)
    clf.fit(train.x, train.y.to_numpy().ravel())
    return pd.DataFrame(clf.predict(test.x), columns=["preds"])


def main():
    """This function runs the SVM model as a standalone program"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    hid_layers, activation_str = interface.remaining_args()
    activation: ActivationType = ACTIVATIONS[activation_str]
    interface.save_predictions(
        train_and_predict(train, test, hid_layers=make_tuple(hid_layers), activation=activation)
    )


if __name__ == "__main__":
    main()
