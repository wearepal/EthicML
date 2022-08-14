"""Implementation of VFAE."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from joblib import dump, load
import pandas as pd
import torch
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from ethicml.data.dataset import CSVDataset
from ethicml.data.lookup import get_dataset_obj_by_name
from ethicml.implementations.beutel import set_seed
from ethicml.utility import DataTuple, SubgroupTuple

from .pytorch_common import CustomDataset, TestDataset
from .utils import load_data_from_flags, save_transformations
from .vfae_modules import VFAENetwork, loss_function

if TYPE_CHECKING:
    from ethicml.models.preprocess.pre_subprocess import PreAlgoArgs, T
    from ethicml.models.preprocess.vfae import VfaeArgs


def fit(train: DataTuple, flags: VfaeArgs):
    """Train the model."""
    dataset = get_dataset_obj_by_name(flags["dataset"])()
    assert isinstance(dataset, CSVDataset)

    # Set up the data
    train_data = CustomDataset(train)
    train_loader = DataLoader(train_data, batch_size=flags["batch_size"])

    # Build Network
    model = VFAENetwork(
        dataset,
        flags["supervised"],
        train_data.xdim,
        latent_dims=flags["latent_dims"],
        z1_enc_size=flags["z1_enc_size"],
        z2_enc_size=flags["z2_enc_size"],
        z1_dec_size=flags["z1_dec_size"],
    ).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run Network
    for epoch in range(int(flags["epochs"])):
        train_model(epoch, model, train_loader, optimizer, flags)
    return model


def transform(model: VFAENetwork, dataset: T, flags: VfaeArgs) -> T:
    """Transform the dataset."""
    data: CustomDataset | TestDataset
    if isinstance(dataset, DataTuple):
        data = CustomDataset(dataset)
        loader = DataLoader(data, batch_size=flags["batch_size"], shuffle=False)
    elif isinstance(dataset, SubgroupTuple):
        data = TestDataset(dataset)
        loader = DataLoader(data, batch_size=flags["batch_size"], shuffle=False)

    post_train: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for sample in loader:
            if isinstance(dataset, DataTuple):
                _x, _s, _ = sample
            elif isinstance(dataset, SubgroupTuple):
                _x, _s = sample
            z1_mu, z1_logvar = model.encode_z1(_x, _s)
            # z1 = model.reparameterize(z1_mu, z1_logvar)
            post_train += z1_mu.data.tolist()

    return dataset.replace(x=pd.DataFrame(post_train))


def train_and_transform(
    train: DataTuple, test: SubgroupTuple, flags: VfaeArgs
) -> tuple[DataTuple, SubgroupTuple]:
    """Train the model and transform both the train dataset and the test dataset."""
    model = fit(train, flags)

    # Transform output
    return transform(model, train, flags), transform(model, test, flags)


def train_model(
    epoch: int, model: VFAENetwork, train_loader: DataLoader, optimizer: Adam, flags: VfaeArgs
) -> None:
    """Train the model."""
    model.train()
    train_loss = 0.0
    for batch_idx, (data_x, data_s, data_y) in enumerate(train_loader):
        data_x = data_x.to("cpu")
        data_s = data_s.to("cpu")
        data_y = data_y.to("cpu")
        optimizer.zero_grad()
        z1_trip, z2_trip, z1_d_trip, x_dec, y_pred = model(data_x, data_s, data_y)

        data_trip = (data_x, data_s, data_y)

        loss_tuple = loss_function(flags, z1_trip, z2_trip, z1_d_trip, data_trip, x_dec, y_pred)
        prediction_loss, reconstruction_loss, kld_loss, mmd_loss = loss_tuple
        loss = kld_loss + reconstruction_loss + prediction_loss + mmd_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            if flags["supervised"]:
                print(
                    f"train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}"
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item() / len(data_x):.6f}\t"
                    f"pred_loss: {prediction_loss.item():.6f}\t"
                    f"recon_loss: {reconstruction_loss.item():.6f}\t"
                    f"kld_loss: {kld_loss.item():.6f}\t"
                    f"mmd_loss: {flags['batch_size'] * mmd_loss.item():.6f}"
                )
            else:
                print(
                    f"train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}"
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item() / len(data_x):.6f}\t"
                    f"recon_loss: {reconstruction_loss.item():.6f}\t"
                    f"mmd_loss: {flags['batch_size'] * mmd_loss.item():.6f}"
                )

    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")


def main() -> None:
    """Run model."""
    pre_algo_args: PreAlgoArgs = json.loads(sys.argv[1])
    flags: VfaeArgs = json.loads(sys.argv[2])
    if pre_algo_args["mode"] == "run":
        set_seed(pre_algo_args["seed"])
        train, test = load_data_from_flags(pre_algo_args)
        save_transformations(train_and_transform(train, test, flags), pre_algo_args)
    elif pre_algo_args["mode"] == "fit":
        set_seed(pre_algo_args["seed"])
        train = DataTuple.from_file(Path(pre_algo_args["train"]))
        enc = fit(train, flags)
        transformed_train = transform(enc, train, flags)
        transformed_train.save_to_file(Path(pre_algo_args["new_train"]))
        dump(enc, Path(pre_algo_args["model"]))
    elif pre_algo_args["mode"] == "transform":
        model = load(Path(pre_algo_args["model"]))
        test = SubgroupTuple.from_file(Path(pre_algo_args["test"]))
        transformed_test = transform(model, test, flags)
        transformed_test.save_to_file(Path(pre_algo_args["new_test"]))


if __name__ == "__main__":
    main()
