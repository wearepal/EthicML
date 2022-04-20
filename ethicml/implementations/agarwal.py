"""Implementation of logistic regression (actually just a wrapper around sklearn)."""
from __future__ import annotations

import contextlib
import json
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

from ethicml.algorithms.inprocess.svm import select_svm
from ethicml.utility import (
    ClassifierType,
    DataTuple,
    FairnessType,
    KernelType,
    Prediction,
    TestTuple,
)

if TYPE_CHECKING:
    from fairlearn.reductions import ExponentiatedGradient

    from ethicml.algorithms.inprocess.agarwal_reductions import AgarwalArgs
    from ethicml.algorithms.inprocess.in_algorithm import InAlgoArgs


def fit(train: DataTuple, args: AgarwalArgs) -> ExponentiatedGradient:
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
    fairness_type = FairnessType[args["fairness"]]
    classifier_type = ClassifierType[args["classifier"]]
    kernel_type = None if args["kernel"] == "" else KernelType[args["kernel"]]

    if fairness_type is FairnessType.dp:
        fairness_class = DemographicParity()
    else:
        fairness_class = EqualizedOdds()

    if classifier_type is ClassifierType.svm:
        model = select_svm(C=args["C"], kernel=kernel_type, seed=args["seed"])
    else:
        random_state = np.random.RandomState(seed=args["seed"])
        model = LogisticRegression(
            solver="liblinear", random_state=random_state, max_iter=5000, C=args["C"]
        )

    data_x = train.x
    data_y = train.y[train.y.columns[0]]
    data_a = train.s[train.s.columns[0]]

    exponentiated_gradient = ExponentiatedGradient(
        model, constraints=fairness_class, eps=args["eps"], T=args["iters"]
    )
    exponentiated_gradient.fit(data_x, data_y, sensitive_features=data_a)

    min_class_label = train.y[train.y.columns[0]].min()
    exponentiated_gradient.min_class_label = min_class_label

    return exponentiated_gradient


def predict(exponentiated_gradient: ExponentiatedGradient, test: TestTuple) -> pd.DataFrame:
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
def working_dir(root: Path) -> Generator[None, None, None]:
    """Change the working directory to the given path."""
    curdir = os.getcwd()
    os.chdir(root.expanduser().resolve().parent)
    try:
        yield
    finally:
        os.chdir(curdir)


def main() -> None:
    """This function runs the Agarwal model as a standalone program."""
    in_algo_args: InAlgoArgs = json.loads(sys.argv[1])
    flags: AgarwalArgs = json.loads(sys.argv[2])
    random.seed(flags["seed"])
    np.random.seed(flags["seed"])
    try:
        import cloudpickle

        # Need to install cloudpickle for now. See https://github.com/fairlearn/fairlearn/issues/569
    except ImportError as e:
        raise RuntimeError("In order to use Agarwal, install fairlearn and cloudpickle.") from e

    if in_algo_args["mode"] == "run":
        train, test = DataTuple.from_npz(Path(in_algo_args["train"])), TestTuple.from_npz(
            Path(in_algo_args["test"])
        )
        Prediction(hard=train_and_predict(train, test, flags)["preds"]).to_npz(
            Path(in_algo_args["predictions"])
        )
    elif in_algo_args["mode"] == "fit":
        data = DataTuple.from_npz(Path(in_algo_args["train"]))
        model = fit(data, flags)
        with working_dir(Path(in_algo_args["model"])):
            model_file = cloudpickle.dumps(model)
        dump(model_file, Path(in_algo_args["model"]))
    elif in_algo_args["mode"] == "predict":
        data = TestTuple.from_npz(Path(in_algo_args["test"]))
        model_file = load(Path(in_algo_args["model"]))
        with working_dir(Path(in_algo_args["model"])):
            model = cloudpickle.loads(model_file)
        Prediction(hard=predict(model, data)["preds"]).to_npz(Path(in_algo_args["predictions"]))
    else:
        raise RuntimeError(f"Unknown mode: {in_algo_args['mode']}")


if __name__ == "__main__":
    main()
