"""
Implementation of VFAE
"""
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from ethicml.data import Adult, Compas, Credit, German, NonBinaryToy, Sqf, Toy
from ethicml.data.dataset import Dataset
from ethicml.implementations.pytorch_common import CustomDataset, TestDataset
from ethicml.implementations.utils import (
    pre_algo_argparser,
    load_data_from_flags,
    save_transformations,
)
from ethicml.implementations.vfae_modules.utils import loss_function
from ethicml.implementations.vfae_modules.vfae_network import VFAENetwork
from ethicml.utility.data_structures import DataTuple, TestTuple


def get_dataset_obj_by_name(name: str) -> Dataset:
    """
    Given a dataset name, get the corresponding dataset object
    Args:
        name:

    Returns:

    """
    lookup: Dict[str, Dataset] = {
        "Adult": Adult(),
        "Compas": Compas(),
        "Credit": Credit(),
        "German": German(),
        "NonBinaryToy": NonBinaryToy(),
        "SQF": Sqf(),
        "Toy": Toy(),
    }

    if name not in lookup:
        raise NotImplementedError("That dataset doesn't exist")

    return lookup[name]


def train_and_transform(
    train: DataTuple, test: TestTuple, flags: Any
) -> Tuple[DataTuple, TestTuple]:
    """
    train the model and transform the dataset
    Args:
        train:
        test:
        flags:

    Returns:
    """
    dataset = get_dataset_obj_by_name(flags['dataset'])

    # Set up the data
    train_data = CustomDataset(train)
    train_loader = DataLoader(train_data, batch_size=flags['batch_size'])

    test_data = TestDataset(test)
    test_loader = DataLoader(test_data, batch_size=flags['batch_size'])

    # Build Network
    model = VFAENetwork(
        dataset,
        flags['supervised'],
        train_data.size,
        latent_dims=50,
        z1_enc_size=flags['z1_enc_size'],
        z2_enc_size=flags['z2_enc_size'],
        z1_dec_size=flags['z1_dec_size'],
    ).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run Network
    for epoch in range(int(flags['epochs'])):
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


def train_model(epoch, model, train_loader, optimizer, flags):
    """
    Train the model
    Args:
        epoch:
        model:
        train_loader:
        optimizer:
        flags:

    Returns:

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
        prediction_loss, reconsruction_loss, kld_loss, mmd_loss = loss_tuple
        loss = kld_loss + reconsruction_loss + prediction_loss + mmd_loss
        loss.backward()
        # KLD_loss = KLD1 + KLD2
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            if flags['supervised']:
                print(
                    f'train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}'
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                    f'Loss: {loss.item() / len(data_x):.6f}\t'
                    f'pred_loss: {prediction_loss.item():.6f}\t'
                    f'recon_loss: {reconsruction_loss.item():.6f}\t'
                    f'kld_loss: {kld_loss.item():.6f}\t'
                    f'mmd_loss: {flags["batch_size"] * mmd_loss.item():.6f}'
                )
            else:
                print(
                    f'train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}'
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                    f'Loss: {loss.item() / len(data_x):.6f}\t'
                    f'recon_loss: {reconsruction_loss.item():.6f}\t'
                    f'mmd_loss: {flags["batch_size"] * mmd_loss.item():.6f}'
                )

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


def main():
    """main method to run model"""
    parser = pre_algo_argparser()

    parser.add_argument("--supervised", type=bool, required=True)
    parser.add_argument("--fairness", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--z1_enc_size", type=int, nargs="+", required=True)
    parser.add_argument("--z2_enc_size", type=int, nargs="+", required=True)
    parser.add_argument("--z1_dec_size", type=int, nargs="+", required=True)
    args = parser.parse_args()

    flags = vars(parser.parse_args())

    train, test = load_data_from_flags(flags)
    save_transformations(train_and_transform(train, test, flags), args)


if __name__ == "__main__":
    main()
