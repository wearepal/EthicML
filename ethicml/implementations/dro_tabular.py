"""Implementation of Fairness without Demographics."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from joblib import dump, load
import numpy as np

try:
    import torch
    from torch import optim
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader

except ImportError as e:
    raise RuntimeError(
        "In order to use PyTorch, please install it following the instructions at https://pytorch.org/ . "
    ) from e


from joblib import dump, load

from ethicml.implementations.beutel import set_seed
from ethicml.implementations.dro_modules.dro_classifier import DROClassifier
from ethicml.implementations.pytorch_common import CustomDataset, TestDataset
from ethicml.implementations.utils import load_data_from_flags
from ethicml.utility import DataTuple, SoftPrediction, SubgroupTuple, TestTuple

if TYPE_CHECKING:
    from ethicml.models.inprocess.fairness_wo_demographics import DroArgs
    from ethicml.models.inprocess.in_subprocess import InAlgoArgs


def train_model(
    epoch: int, model: DROClassifier, train_loader: DataLoader, optimizer: Optimizer
) -> None:
    """Train a model."""
    model.train()
    train_loss = 0.0
    for batch_idx, (data_x, _, data_y) in enumerate(train_loader):
        data_x = data_x.to("cpu")
        data_y = data_y.to("cpu")
        optimizer.zero_grad()
        y_prob = model.forward(data_x)
        loss = model.loss(y_prob.squeeze(-1), data_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}"
                f"\t({100. * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item() / len(data_x):.6f}"
            )

    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")


def fit(train: DataTuple, args: DroArgs, seed: int) -> DROClassifier:
    """Train a network and return predictions."""
    # Set up the data
    set_seed(seed)
    train_data = CustomDataset(train)
    train_loader = DataLoader(train_data, batch_size=args["batch_size"])

    # Build Network
    model = DROClassifier(
        in_size=train_data.xdim,
        out_size=train_data.ydim,
        network_size=args["network_size"],
        eta=args["eta"],
    ).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run Network
    for epoch in range(int(args["epochs"])):
        train_model(epoch, model, train_loader, optimizer)
    return model


def predict(model: DROClassifier, test: TestTuple, args: DroArgs) -> SoftPrediction:
    """Train a network and return predictions."""
    # Set up the data
    test_data = TestDataset(test)
    test_loader = DataLoader(test_data, batch_size=args["batch_size"])

    # Transform output
    post_test: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for _x, _ in test_loader:
            post_test += model.forward(_x)

    return SoftPrediction(soft=np.array([j.softmax(dim=-1).numpy() for j in post_test]))


def train_and_predict(
    train: DataTuple, test: TestTuple, args: DroArgs, seed: int
) -> SoftPrediction:
    """Train a network and return predictions."""
    # Set up the data
    set_seed(seed)
    train_data = CustomDataset(train)
    train_loader = DataLoader(train_data, batch_size=args["batch_size"])

    test_data = TestDataset(test)
    test_loader = DataLoader(test_data, batch_size=args["batch_size"])

    # Build Network
    model = DROClassifier(
        in_size=train_data.xdim,
        out_size=train_data.ydim,
        network_size=args["network_size"],
        eta=args["eta"],
    ).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run Network
    for epoch in range(int(args["epochs"])):
        train_model(epoch, model, train_loader, optimizer)

    # Transform output
    post_test: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for _x, _ in test_loader:
            post_test += model.forward(_x)

    return SoftPrediction(soft=np.array([j.softmax(dim=-1).numpy() for j in post_test]))


def main() -> None:
    """Run the FWD model as a standalone program on tabular data."""
    in_algo_args: InAlgoArgs = json.loads(sys.argv[1])
    flags: DroArgs = json.loads(sys.argv[2])
    data: DataTuple | TestTuple
    if in_algo_args["mode"] == "run":
        train, test = load_data_from_flags(in_algo_args)
        train_and_predict(train, test, flags, seed=in_algo_args["seed"]).save_to_file(
            Path(in_algo_args["predictions"])
        )
    elif in_algo_args["mode"] == "fit":
        data = DataTuple.from_file(Path(in_algo_args["train"]))
        model = fit(data, flags, seed=in_algo_args["seed"])
        dump(model, Path(in_algo_args["model"]))
    elif in_algo_args["mode"] == "predict":
        data = SubgroupTuple.from_file(Path(in_algo_args["test"]))
        model = load(Path(in_algo_args["model"]))
        predict(model, data, flags).save_to_file(Path(in_algo_args["predictions"]))


if __name__ == "__main__":
    main()
