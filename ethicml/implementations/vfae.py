"""Implementation of VFAE."""
from typing import List, Tuple

import pandas as pd
import torch
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from ethicml.data.lookup import get_dataset_obj_by_name
from ethicml.utility import DataTuple, TestTuple

from .pytorch_common import CustomDataset, TestDataset
from .utils import PreAlgoArgs, load_data_from_flags, save_transformations
from .vfae_modules import VFAENetwork, loss_function


class VfaeArgs(PreAlgoArgs):
    """Args object of VFAE."""

    supervised: bool
    fairness: str
    batch_size: int
    epochs: int
    dataset: str
    z1_enc_size: List[int]
    z2_enc_size: List[int]
    z1_dec_size: List[int]


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
    dataset = get_dataset_obj_by_name(flags.dataset)()

    # Set up the data
    train_data = CustomDataset(train)
    train_loader = DataLoader(train_data, batch_size=flags.batch_size)

    test_data = TestDataset(test)
    test_loader = DataLoader(test_data, batch_size=flags.batch_size)

    # Build Network
    model = VFAENetwork(
        dataset,
        flags.supervised,
        train_data.xdim,
        latent_dims=50,
        z1_enc_size=flags.z1_enc_size,
        z2_enc_size=flags.z2_enc_size,
        z1_dec_size=flags.z1_dec_size,
    ).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run Network
    for epoch in range(int(flags.epochs)):
        train_model(epoch, model, train_loader, optimizer, flags)

    # Transform output
    post_train: List[List[float]] = []
    post_test: List[List[float]] = []
    model.eval()
    with torch.no_grad():
        for _x, _s, _ in train_loader:
            z1_mu, z1_logvar = model.encode_z1(_x, _s)
            z1 = model.reparameterize(z1_mu, z1_logvar)
            post_train += z1.data.tolist()
        for _x, _s in test_loader:
            z1_mu, z1_logvar = model.encode_z1(_x, _s)
            z1 = model.reparameterize(z1_mu, z1_logvar)
            post_test += z1.data.tolist()

    return (
        DataTuple(x=pd.DataFrame(post_train), s=train.s, y=train.y, name=f"VFAE: {train.name}"),
        TestTuple(x=pd.DataFrame(post_test), s=test.s, name=f"VFAE: {test.name}"),
    )


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
    train_loss = 0
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
        # KLD_loss = KLD1 + KLD2
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


def main():
    """Main method to run model."""
    args = VfaeArgs(explicit_bool=True).parse_args()
    train, test = load_data_from_flags(args)
    save_transformations(train_and_transform(train, test, args), args)


if __name__ == "__main__":
    main()
