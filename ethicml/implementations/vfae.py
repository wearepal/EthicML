"""Implementation of VFAE."""
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch
from joblib import dump, load
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from ethicml.algorithms.preprocess.pre_algorithm import T
from ethicml.data.lookup import get_dataset_obj_by_name
from ethicml.implementations.beutel import set_seed
from ethicml.utility import DataTuple, TestTuple

from .pytorch_common import CustomDataset, TestDataset
from .utils import load_data_from_flags, save_transformations
from .vfae_modules import VfaeArgs, VFAENetwork, loss_function


def fit(train, flags):
    """Train the model."""
    dataset = get_dataset_obj_by_name(flags.dataset)()

    # Set up the data
    train_data = CustomDataset(train)
    train_loader = DataLoader(train_data, batch_size=flags.batch_size)

    # Build Network
    model = VFAENetwork(
        dataset,
        flags.supervised,
        train_data.xdim,
        latent_dims=flags.latent_dims,
        z1_enc_size=flags.z1_enc_size,
        z2_enc_size=flags.z2_enc_size,
        z1_dec_size=flags.z1_dec_size,
    ).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run Network
    for epoch in range(int(flags.epochs)):
        train_model(epoch, model, train_loader, optimizer, flags)
    return model


def transform(model: VFAENetwork, dataset: T, flags) -> T:
    """Transform the dataset."""
    data: Union[CustomDataset, TestDataset]
    if isinstance(dataset, DataTuple):
        data = CustomDataset(dataset)
        loader = DataLoader(data, batch_size=flags.batch_size, shuffle=False)
    elif isinstance(dataset, TestTuple):
        data = TestDataset(dataset)
        loader = DataLoader(data, batch_size=flags.batch_size, shuffle=False)

    post_train: List[List[float]] = []
    model.eval()
    with torch.no_grad():
        for sample in loader:
            if isinstance(dataset, DataTuple):
                _x, _s, _ = sample
            elif isinstance(dataset, TestTuple):
                _x, _s = sample
            z1_mu, z1_logvar = model.encode_z1(_x, _s)
            # z1 = model.reparameterize(z1_mu, z1_logvar)
            post_train += z1_mu.data.tolist()

    if isinstance(dataset, DataTuple):
        return DataTuple(
            x=pd.DataFrame(post_train), s=dataset.s, y=dataset.y, name=f"VFAE: {dataset.name}"
        )
    elif isinstance(dataset, TestTuple):
        return TestTuple(x=pd.DataFrame(post_train), s=dataset.s, name=f"VFAE: {dataset.name}")


def train_and_transform(
    train: DataTuple, test: TestTuple, flags: VfaeArgs
) -> Tuple[DataTuple, TestTuple]:
    """Train the model and transform the dataset.

    Args:
        train:
        test:
        flags:

    Returns:
        Tuple of Encoded Train Dataset and Test Dataset.
    """
    model = fit(train, flags)

    # Transform output
    return transform(model, train, flags), transform(model, test, flags)


def train_model(
    epoch: int, model: VFAENetwork, train_loader: DataLoader, optimizer: Adam, flags: VfaeArgs
) -> None:
    """Train the model.

    Args:
        epoch:
        model:
        train_loader:
        optimizer:
        flags:

    Returns:
        None
    """
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
            if flags.supervised:
                print(
                    f"train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}"
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item() / len(data_x):.6f}\t"
                    f"pred_loss: {prediction_loss.item():.6f}\t"
                    f"recon_loss: {reconstruction_loss.item():.6f}\t"
                    f"kld_loss: {kld_loss.item():.6f}\t"
                    f"mmd_loss: {flags.batch_size * mmd_loss.item():.6f}"
                )
            else:
                print(
                    f"train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}"
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item() / len(data_x):.6f}\t"
                    f"recon_loss: {reconstruction_loss.item():.6f}\t"
                    f"mmd_loss: {flags.batch_size * mmd_loss.item():.6f}"
                )

    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")


def main() -> None:
    """Main method to run model."""
    args = VfaeArgs(explicit_bool=True).parse_args()
    set_seed(args.seed)
    if args.mode == "run":
        assert args.train is not None
        assert args.new_train is not None
        assert args.test is not None
        assert args.new_test is not None
        train, test = load_data_from_flags(args)
        save_transformations(train_and_transform(train, test, args), args)
    elif args.mode == "fit":
        assert args.model is not None
        assert args.train is not None
        assert args.new_train is not None
        train = DataTuple.from_npz(Path(args.train))
        enc = fit(train, args)
        transformed_train = transform(enc, train, args)
        transformed_train.to_npz(Path(args.new_train))
        dump(enc, Path(args.model))
    elif args.mode == "transform":
        assert args.model is not None
        assert args.test is not None
        assert args.new_test is not None
        test = DataTuple.from_npz(Path(args.test))
        model = load(Path(args.model))
        transformed_test = transform(model, test, args)
        transformed_test.to_npz(Path(args.new_test))


if __name__ == "__main__":
    main()
