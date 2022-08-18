"""Fair Dummies Implementation."""
from __future__ import annotations
import json
from pathlib import Path
import random
import sys
from typing import TYPE_CHECKING

from joblib import dump, load
import numpy as np

from ethicml import DataTuple, SoftPrediction, SubgroupTuple
from ethicml.implementations.hgr_modules.hgr_impl import HgrClassLearner
from ethicml.utility.data_structures import ModelType

if TYPE_CHECKING:
    from ethicml.models.inprocess.hgr import HgrArgs
    from ethicml.models.inprocess.in_subprocess import InAlgoArgs


def fit(train: DataTuple, args: HgrArgs, seed: int = 888) -> HgrClassLearner:
    """Fit a model."""
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError as e:
        raise RuntimeError(
            "In order to use PyTorch, please install it following the instructions as https://pytorch.org/ . "
        ) from e

    random.seed(seed)
    np.random.seed(seed)

    model = HgrClassLearner(
        lr=args["lr"],
        epochs=args["epochs"],
        mu=args["mu"],
        cost_pred=torch.nn.CrossEntropyLoss(),
        in_shape=len(train.x.columns),
        out_shape=train.y.nunique(),
        batch_size=args["batch_size"],
        model_type=ModelType(args["model_type"]),
    )
    return model.fit(train, seed=seed)


def predict(model: HgrClassLearner, test: SubgroupTuple) -> np.ndarray:
    """Compute predictions on the given test data."""
    return model.predict(test.x)


def train_and_predict(
    train: DataTuple, test: SubgroupTuple, args: HgrArgs, seed: int
) -> np.ndarray:
    """Train a logistic regression model and compute predictions on the given test data."""
    model = fit(train, args, seed)
    return predict(model, test)


def main() -> None:
    """Run the Agarwal model as a standalone program."""
    in_algo_args: InAlgoArgs = json.loads(sys.argv[1])
    flags: HgrArgs = json.loads(sys.argv[2])

    if in_algo_args["mode"] == "run":
        train, test = DataTuple.from_file(Path(in_algo_args["train"])), SubgroupTuple.from_file(
            Path(in_algo_args["test"])
        )
        SoftPrediction(
            soft=train_and_predict(train, test, flags, in_algo_args["seed"])
        ).save_to_file(Path(in_algo_args["predictions"]))
    elif in_algo_args["mode"] == "fit":
        data = DataTuple.from_file(Path(in_algo_args["train"]))
        model = fit(data, flags, in_algo_args["seed"])
        setattr(model, "ethicml_random_seed", in_algo_args["seed"])  # need to save the seed as well
        dump(model, Path(in_algo_args["model"]))
    elif in_algo_args["mode"] == "predict":
        test = SubgroupTuple.from_file(Path(in_algo_args["test"]))
        model = load(Path(in_algo_args["model"]))
        SoftPrediction(soft=predict(model, test)).save_to_file(Path(in_algo_args["predictions"]))
    else:
        raise RuntimeError(f"Unknown mode: {in_algo_args['mode']}")


if __name__ == "__main__":
    main()
