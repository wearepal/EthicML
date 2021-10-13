"""Implementation of logistic regression (actually just a wrapper around sklearn)."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ethicml.algorithms.inprocess.svm import select_svm
from ethicml.utility import ClassifierType, DataTuple, FairnessType, Prediction, TestTuple

from .utils import InAlgoArgs


class AgarwalArgs(InAlgoArgs):
    """Args for the Agarwal implementation."""

    classifier: ClassifierType
    fairness: FairnessType
    eps: float
    iters: int
    C: float
    kernel: str
    seed: int


def train_and_predict(train: DataTuple, test: TestTuple, args: AgarwalArgs) -> pd.DataFrame:
    """Train a logistic regression model and compute predictions on the given test data."""
    try:
        from fairlearn.reductions import (
            ConditionalSelectionRate,
            DemographicParity,
            EqualizedOdds,
            ExponentiatedGradient,
        )
    except ImportError as e:
        raise RuntimeError("In order to use Agarwal, install fairlearn.") from e

    random.seed(args.seed)
    np.random.seed(args.seed)

    fairness_class: ConditionalSelectionRate
    if args.fairness == "DP":
        fairness_class = DemographicParity()
    else:
        fairness_class = EqualizedOdds()

    if args.classifier == "SVM":
        model = select_svm(C=args.C, kernel=args.kernel, seed=args.seed)
    else:
        random_state = np.random.RandomState(seed=args.seed)
        model = LogisticRegression(
            solver="liblinear", random_state=random_state, max_iter=5000, C=args.C
        )

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


def main() -> None:
    """This function runs the Agarwal model as a standalone program."""
    args: AgarwalArgs = AgarwalArgs().parse_args()
    train, test = DataTuple.from_npz(Path(args.train)), TestTuple.from_npz(Path(args.test))
    Prediction(hard=train_and_predict(train, test, args)["preds"]).to_npz(Path(args.predictions))


if __name__ == "__main__":
    main()
