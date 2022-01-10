"""Implementation of logistic regression (actually just a wrapper around sklearn)."""
import contextlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
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


def fit(train: DataTuple, args):
    """Fit a model."""
    try:
        from fairlearn.reductions import (
            ConditionalSelectionRate,
            DemographicParity,
            EqualizedOdds,
            ExponentiatedGradient,
        )
    except ImportError as e:
        raise RuntimeError("In order to use Agarwal, install fairlearn==0.4.6.") from e

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

    min_class_label = train.y[train.y.columns[0]].min()
    exponentiated_gradient.min_class_label = min_class_label

    return exponentiated_gradient


def predict(exponentiated_gradient, test: TestTuple) -> pd.DataFrame:
    """Compute predictions on the given test data."""
    randomized_predictions = exponentiated_gradient.predict(test.x)
    preds = pd.DataFrame(randomized_predictions, columns=["preds"])

    if preds["preds"].min() != preds["preds"].max():
        preds = preds.replace(preds["preds"].min(), exponentiated_gradient.min_class_label)
    return preds


def train_and_predict(train: DataTuple, test: TestTuple, args: AgarwalArgs) -> pd.DataFrame:
    """Train a logistic regression model and compute predictions on the given test data."""
    exponentiated_gradient = fit(train, args)
    return predict(exponentiated_gradient, test)


@contextlib.contextmanager
def working_dir(root: Path) -> None:
    """Change the working directory to the given path."""
    curdir = os.getcwd()
    os.chdir(root.expanduser().resolve().parent)
    try:
        yield
    finally:
        os.chdir(curdir)


def main() -> None:
    """This function runs the Agarwal model as a standalone program."""
    args: AgarwalArgs = AgarwalArgs().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import cloudpickle

        # Need to install cloudpickle for now. See https://github.com/fairlearn/fairlearn/issues/569
    except ImportError as e:
        raise RuntimeError("In order to use Agarwal, install fairlearn and cloudpickle.") from e

    if args.mode == "run":
        assert args.train is not None
        assert args.test is not None
        assert args.predictions is not None
        train, test = DataTuple.from_npz(Path(args.train)), TestTuple.from_npz(Path(args.test))
        Prediction(hard=train_and_predict(train, test, args)["preds"]).to_npz(
            Path(args.predictions)
        )
    elif args.mode == "fit":
        assert args.train is not None
        assert args.model is not None
        data = DataTuple.from_npz(Path(args.train))
        model = fit(data, args)
        with working_dir(Path(args.model)):
            model_file = cloudpickle.dumps(model)
        dump(model_file, Path(args.model))
    elif args.mode == "predict":
        assert args.model is not None
        assert args.predictions is not None
        assert args.test is not None
        data = TestTuple.from_npz(Path(args.test))
        model_file = load(Path(args.model))
        with working_dir(Path(args.model)):
            model = cloudpickle.loads(model_file)
        Prediction(hard=predict(model, data)["preds"]).to_npz(Path(args.predictions))
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
