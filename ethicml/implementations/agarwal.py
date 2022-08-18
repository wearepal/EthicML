"""Implementation of logistic regression (actually just a wrapper around sklearn)."""
from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import random
import sys
from typing import TYPE_CHECKING, Generator

from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ethicml.models.inprocess.svm import select_svm
from ethicml.utility import (
    ClassifierType,
    DataTuple,
    FairnessType,
    KernelType,
    Prediction,
    SubgroupTuple,
    TestTuple,
)

if TYPE_CHECKING:
    from fairlearn.reductions import ExponentiatedGradient

    from ethicml.models.inprocess.agarwal_reductions import AgarwalArgs
    from ethicml.models.inprocess.in_subprocess import InAlgoArgs


def fit(train: DataTuple, args: AgarwalArgs, seed: int = 888) -> ExponentiatedGradient:
    """Fit a model."""
    try:
        from fairlearn.reductions import (
            DemographicParity,
            EqualizedOdds,
            ExponentiatedGradient,
            UtilityParity,
        )
    except ImportError as e:
        raise RuntimeError(
            f"In order to use Agarwal, install fairlearn==0.7.0. "
            f"Consider installing EthicML with the extras 'all' specified."
        ) from e

    fairness_class: UtilityParity
    fairness_type = FairnessType(args["fairness"])
    classifier_type = ClassifierType(args["classifier"])
    kernel_type = None if args["kernel"] == "" else KernelType[args["kernel"]]

    if fairness_type is FairnessType.dp:
        fairness_class = DemographicParity(difference_bound=args["eps"])
    else:
        fairness_class = EqualizedOdds(difference_bound=args["eps"])

    if classifier_type is ClassifierType.svm:
        assert kernel_type is not None
        model = select_svm(C=args["C"], kernel=kernel_type, seed=seed)
    elif classifier_type is ClassifierType.lr:
        random_state = np.random.RandomState(seed=seed)
        model = LogisticRegression(
            solver="liblinear", random_state=random_state, max_iter=5000, C=args["C"]
        )
    elif classifier_type is ClassifierType.gbt:
        random_state = np.random.RandomState(seed=seed)
        model = GradientBoostingClassifier(random_state=random_state)

    data_x = train.x
    data_y = train.y
    data_a = train.s

    exponentiated_gradient = ExponentiatedGradient(
        model, constraints=fairness_class, eps=args["eps"], max_iter=args["iters"]
    )
    exponentiated_gradient.fit(data_x, data_y, sensitive_features=data_a)

    min_class_label = train.y.min()
    exponentiated_gradient.min_class_label = min_class_label

    return exponentiated_gradient


def predict(exponentiated_gradient: ExponentiatedGradient, test: TestTuple) -> pd.DataFrame:
    """Compute predictions on the given test data."""
    randomized_predictions = exponentiated_gradient.predict(test.x)
    preds = pd.DataFrame(randomized_predictions, columns=["preds"])

    if preds["preds"].min() != preds["preds"].max():
        preds = preds.replace(preds["preds"].min(), exponentiated_gradient.min_class_label)
    return preds


def train_and_predict(
    train: DataTuple, test: TestTuple, args: AgarwalArgs, seed: int
) -> pd.DataFrame:
    """Train a logistic regression model and compute predictions on the given test data."""
    exponentiated_gradient = fit(train, args, seed)
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
    """Run the Agarwal model as a standalone program."""
    in_algo_args: InAlgoArgs = json.loads(sys.argv[1])
    flags: AgarwalArgs = json.loads(sys.argv[2])
    try:
        import cloudpickle

        # Need to install cloudpickle for now. See https://github.com/fairlearn/fairlearn/issues/569
    except ImportError as e:
        raise RuntimeError("In order to use Agarwal, install fairlearn and cloudpickle.") from e

    if in_algo_args["mode"] == "run":
        random.seed(in_algo_args["seed"])
        np.random.seed(in_algo_args["seed"])
        train, test = DataTuple.from_file(Path(in_algo_args["train"])), SubgroupTuple.from_file(
            Path(in_algo_args["test"])
        )
        Prediction(
            hard=train_and_predict(train, test, flags, in_algo_args["seed"])["preds"]
        ).save_to_file(Path(in_algo_args["predictions"]))
    elif in_algo_args["mode"] == "fit":
        random.seed(in_algo_args["seed"])
        np.random.seed(in_algo_args["seed"])
        data = DataTuple.from_file(Path(in_algo_args["train"]))
        model = fit(data, flags, in_algo_args["seed"])
        with working_dir(Path(in_algo_args["model"])):
            model.ethicml_random_seed = in_algo_args["seed"]  # need to save the seed as well
            model_file = cloudpickle.dumps(model)
        dump(model_file, Path(in_algo_args["model"]))
    elif in_algo_args["mode"] == "predict":
        testdata = SubgroupTuple.from_file(Path(in_algo_args["test"]))
        model_file = load(Path(in_algo_args["model"]))
        with working_dir(Path(in_algo_args["model"])):
            model = cloudpickle.loads(model_file)
            seed = model.ethicml_random_seed
        random.seed(seed)
        np.random.seed(seed)
        Prediction(hard=predict(model, testdata)["preds"]).save_to_file(
            Path(in_algo_args["predictions"])
        )
    else:
        raise RuntimeError(f"Unknown mode: {in_algo_args['mode']}")


if __name__ == "__main__":
    main()
