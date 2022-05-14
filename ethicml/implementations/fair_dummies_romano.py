"""Fair Dummies Implementation."""
import json
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from joblib import dump, load

from ethicml import DataTuple, SoftPrediction, TestTuple
from ethicml.implementations.fair_dummies_modules.model import EquiClassLearner

if TYPE_CHECKING:
    from ethicml.algorithms.inprocess.fair_dummies import FairDummiesArgs
    from ethicml.algorithms.inprocess.in_subprocess import InAlgoArgs


def fit(train: DataTuple, args: FairDummiesArgs, seed: int = 888) -> EquiClassLearner:
    """Fit a model.

    :param train:
    :param args:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = EquiClassLearner(
        lr=args["lr"],
        pretrain_pred_epochs=args["pretrain_pred_epochs"],
        pretrain_dis_epochs=args["pretrain_dis_epochs"],
        epochs=args["epochs"],
        loss_steps=args["loss_steps"],
        dis_steps=args["dis_steps"],
        cost_pred=torch.nn.CrossEntropyLoss(),
        in_shape=len(train.x.columns),
        batch_size=args["batch_size"],
        model_type=args["model_type"],
        lambda_vec=args["lambda_vec"],
        second_moment_scaling=args["second_moment_scaling"],
        num_classes=train.y.nunique(),
    )
    input_data_train = pd.concat([train.s, train.x], axis="columns").to_numpy()
    model.fit(input_data_train, train.y.to_numpy())
    return model


def predict(model: EquiClassLearner, test: TestTuple) -> pd.DataFrame:
    """Compute predictions on the given test data.

    :param exponentiated_gradient:
    :param test:
    """
    input_data_test = pd.concat([test.s, test.x], axis="columns").to_numpy()
    randomized_predictions = model.predict(input_data_test)
    preds = pd.DataFrame(randomized_predictions[:, 1], columns=["preds"])
    return preds


def train_and_predict(
    train: DataTuple, test: TestTuple, args: FairDummiesArgs, seed: int
) -> pd.DataFrame:
    """Train a logistic regression model and compute predictions on the given test data.

    :param train:
    :param test:
    :param args:
    """
    model = fit(train, args, seed)
    return predict(model, test)


def main() -> None:
    """Run the Agarwal model as a standalone program."""
    in_algo_args: InAlgoArgs = json.loads(sys.argv[1])
    flags: FairDummiesArgs = json.loads(sys.argv[2])

    if in_algo_args["mode"] == "run":
        train, test = DataTuple.from_npz(Path(in_algo_args["train"])), TestTuple.from_npz(
            Path(in_algo_args["test"])
        )
        SoftPrediction(
            soft=train_and_predict(train, test, flags, in_algo_args["seed"])["preds"]
        ).to_npz(Path(in_algo_args["predictions"]))
    elif in_algo_args["mode"] == "fit":
        data = DataTuple.from_npz(Path(in_algo_args["train"]))
        model = fit(data, flags, in_algo_args["seed"])
        model.ethicml_random_seed = in_algo_args["seed"]  # need to save the seed as well
        dump(model, Path(in_algo_args["model"]))
    elif in_algo_args["mode"] == "predict":
        testdata = TestTuple.from_npz(Path(in_algo_args["test"]))
        model = load(Path(in_algo_args["model"]))
        SoftPrediction(soft=predict(model, testdata)["preds"]).to_npz(
            Path(in_algo_args["predictions"])
        )
    else:
        raise RuntimeError(f"Unknown mode: {in_algo_args['mode']}")


if __name__ == "__main__":
    main()
