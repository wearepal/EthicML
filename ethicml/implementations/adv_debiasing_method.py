"""Fair Dummies Implementation."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from joblib import dump, load

from ethicml import DataTuple, SoftPrediction, SubgroupTuple
from ethicml.algorithms.inprocess.adv_debiasing import AdvDebArgs
from ethicml.implementations.adv_debiasing_modules.model import AdvDebiasingClassLearner

if TYPE_CHECKING:
    from ethicml.algorithms.inprocess.in_subprocess import InAlgoArgs


def fit(train: DataTuple, args: AdvDebArgs, seed: int = 888) -> AdvDebiasingClassLearner:
    """Fit a model.

    :param train:
    :param args:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = AdvDebiasingClassLearner(
        lr=args["lr"],
        n_clf_epochs=args["n_clf_epochs"],
        n_adv_epochs=args["n_adv_epochs"],
        n_epoch_combined=args["n_epoch_combined"],
        cost_pred=torch.nn.CrossEntropyLoss(),
        in_shape=len(train.x.columns),
        batch_size=args["batch_size"],
        model_type=args["model_type"],
        num_classes=train.y.nunique(),
        lambda_vec=args["lambda_vec"],
    )
    return model.fit(train, seed=seed)


def predict(model: AdvDebiasingClassLearner, test: SubgroupTuple) -> np.ndarray:
    """Compute predictions on the given test data.

    :param exponentiated_gradient:
    :param test:
    """
    return model.predict(test.x)


def train_and_predict(
    train: DataTuple, test: SubgroupTuple, args: AdvDebArgs, seed: int
) -> np.ndarray:
    """Train a logistic regression model and compute predictions on the given test data.

    :param train:
    :param test:
    :param args:
    """
    model = fit(train, args, seed)
    return predict(model, test)


def main() -> None:
    """Run the adversarial debiasing model as a standalone program."""
    in_algo_args: InAlgoArgs = json.loads(sys.argv[1])
    flags: AdvDebArgs = json.loads(sys.argv[2])

    if in_algo_args["mode"] == "run":
        train, test = DataTuple.from_npz(Path(in_algo_args["train"])), SubgroupTuple.from_npz(
            Path(in_algo_args["test"])
        )
        SoftPrediction(soft=train_and_predict(train, test, flags, in_algo_args["seed"])).to_npz(
            Path(in_algo_args["predictions"])
        )
    elif in_algo_args["mode"] == "fit":
        data = DataTuple.from_npz(Path(in_algo_args["train"]))
        model = fit(data, flags, in_algo_args["seed"])
        setattr(model, "ethicml_random_seed", in_algo_args["seed"])  # need to save the seed as well
        dump(model, Path(in_algo_args["model"]))
    elif in_algo_args["mode"] == "predict":
        test = SubgroupTuple.from_npz(Path(in_algo_args["test"]))
        model = load(Path(in_algo_args["model"]))
        SoftPrediction(soft=predict(model, test)).to_npz(Path(in_algo_args["predictions"]))
    else:
        raise RuntimeError(f"Unknown mode: {in_algo_args['mode']}")


if __name__ == "__main__":
    main()
