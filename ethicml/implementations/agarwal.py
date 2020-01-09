"""Implementation of logistic regression (actually just a wrapper around sklearn)."""
import random

import numpy as np
import pandas as pd

from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity,
    EqualizedOdds,
    ConditionalSelectionRate,
)
from sklearn.linear_model import LogisticRegression

from ethicml.utility.data_structures import DataTuple, TestTuple, FairnessType, str_to_fair_type
from ethicml.implementations.utils import InAlgoInterface
from ethicml.implementations.svm import select_svm


def train_and_predict(
    train: DataTuple,
    test: TestTuple,
    classifier: str,
    fairness: FairnessType,
    eps: float,
    iters: int,
    C: float,
    kernel: str,
):
    """Train a logistic regression model and compute predictions on the given test data."""
    random.seed(888)
    np.random.seed(888)

    fairness_class: ConditionalSelectionRate
    if fairness == "DP":
        fairness_class = DemographicParity()
    else:
        fairness_class = EqualizedOdds()

    if classifier == "SVM":
        model = select_svm(C, kernel)
    else:
        model = LogisticRegression(solver="liblinear", random_state=888, max_iter=5000, C=float(C))

    data_x = train.x
    data_y = train.y[train.y.columns[0]]
    data_a = train.s[train.s.columns[0]]

    exponentiated_gradient = ExponentiatedGradient(
        model, constraints=fairness_class, eps=eps, T=iters
    )
    exponentiated_gradient.fit(data_x, data_y, sensitive_features=data_a)

    randomized_predictions = exponentiated_gradient.predict(test.x)
    preds = pd.DataFrame(randomized_predictions, columns=["preds"])

    min_class_label = train.y[train.y.columns[0]].min()
    if preds["preds"].min() != preds["preds"].max():
        preds = preds.replace(preds["preds"].min(), min_class_label)
    return preds


def main():
    """This function runs the Agarwal model as a standalone program."""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    classifier, fairness_, eps, iters, C, kernel = interface.remaining_args()
    fairness = str_to_fair_type(fairness_)
    assert fairness is not None
    interface.save_predictions(
        train_and_predict(
            train,
            test,
            classifier=classifier,
            fairness=fairness,
            eps=float(eps),
            iters=int(iters),
            C=float(C),
            kernel=kernel,
        )
    )


if __name__ == "__main__":
    main()
