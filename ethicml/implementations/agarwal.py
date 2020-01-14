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

from ethicml.utility.data_structures import DataTuple, TestTuple, FairnessType, ClassifierType
from ethicml.algorithms.inprocess.svm import select_svm
from .utils import InAlgoArgs, load_data_from_flags, save_predictions


class AgarwalArgs(InAlgoArgs):
    """Args for the Agarwal implementation."""

    classifier: ClassifierType
    fairness: FairnessType
    eps: float
    iters: int
    C: float
    kernel: str


def train_and_predict(train: DataTuple, test: TestTuple, args: AgarwalArgs):
    """Train a logistic regression model and compute predictions on the given test data."""
    random.seed(888)
    np.random.seed(888)

    fairness_class: ConditionalSelectionRate
    if args.fairness == "DP":
        fairness_class = DemographicParity()
    else:
        fairness_class = EqualizedOdds()

    if args.classifier == "SVM":
        model = select_svm(args.C, args.kernel)
    else:
        model = LogisticRegression(solver="liblinear", random_state=888, max_iter=5000, C=args.C)

    data_x = train.x
    data_y = train.y[train.y.columns[0]]
    data_a = train.s[train.s.columns[0]]

    exponentiated_gradient = ExponentiatedGradient(
        model, constraints=fairness_class, eps=args.eps, T=args.iters
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
    args: AgarwalArgs = AgarwalArgs().parse_args()
    train, test = load_data_from_flags(args)
    save_predictions(train_and_predict(train, test, args), args)


if __name__ == "__main__":
    main()
