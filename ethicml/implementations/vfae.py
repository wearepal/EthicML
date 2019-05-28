"""
Implementation of VFAE
"""
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.utils.data import DataLoader

from ethicml.algorithms.utils import DataTuple
from ethicml.data import Adult, Compas, Credit, German, NonBinaryToy, Sqf, Toy
from ethicml.data.dataset import Dataset
from ethicml.implementations.pytorch_common import CustomDataset
from ethicml.implementations.vfae_modules.utils import loss_function
from ethicml.implementations.vfae_modules.vfae_network import VFAENetwork


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
        "Toy": Toy()
    }

    if name not in lookup:
        raise NotImplementedError("That dataset doesn't exist")

    return lookup[name]


def random_seed(seed_value, use_cuda=False):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def train_and_transform(train, test, flags):
    """
    train the model and transform the dataset
    Args:
        train:
        test:
        flags:

    Returns:
    """
    random_seed(888)

    dataset = get_dataset_obj_by_name(flags['dataset'])

    # Order dataset and minmax cont feats
    cont_feats = dataset.continuous_features

    if cont_feats:
        scaler = MinMaxScaler()
        train_df = train.x
        train_df[cont_feats] = scaler.fit_transform(train_df[cont_feats])
        train = DataTuple(x=train_df, s=train.s, y=train.y)

        test_df = test.x
        test_df[cont_feats] = scaler.transform(test_df[cont_feats])
        test = DataTuple(x=test_df, s=test.s, y=test.y)

    # Set up the data
    train_data = CustomDataset(train)
    train_loader = DataLoader(train_data, batch_size=flags['batch_size'])

    test_data = CustomDataset(test)
    test_loader = DataLoader(test_data, batch_size=flags['batch_size'])

    # Build Network
    model = VFAENetwork(dataset, flags['supervised'], train_data.size, latent_dims=50,
                        z1_enc_size=flags['z1_enc_size'],
                        z2_enc_size=flags['z2_enc_size'],
                        z1_dec_size=flags['z1_dec_size']).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Run Network
    for epoch in range(flags['epochs']):
        train_model(epoch, model, train_loader, optimizer, flags)

    # Transform output
    post_train: List[List[float]] = []
    post_test: List[List[float]] = []
    for embedding, _s, _y in train_loader:
        model.eval()
        z1_trip, _, _, x_dec, _ = model(embedding, _s, _y)
        # train += scaler.inverse_transform(x_dec.data).tolist()
        # post_train += x_dec.data.tolist()
        post_train += z1_trip[0].data.tolist()
    for embedding, _s, _y in test_loader:
        model.eval()
        z1_trip, _, _, x_dec, _ = model(embedding, _s, _y)
        # test += scaler.inverse_transform(x_dec.data).tolist()
        # post_test += x_dec.data.tolist()
        post_test += z1_trip[0].data.tolist()

    # post_train = pd.DataFrame(post_train, columns=train.x.columns)
    post_train = pd.DataFrame(post_train)
    # post_test = pd.DataFrame(post_test, columns=test.x.columns)
    post_test = pd.DataFrame(post_test)

    # post_train[cont_feats] = scaler.inverse_transform(post_train[cont_feats])
    # post_test[cont_feats] = scaler.inverse_transform(post_test[cont_feats])

    return post_train, post_test


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
        if epoch % 20 == 0 and batch_idx == 0:
            if flags['supervised']:
                print(f'train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}'
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item() / len(data_x):.6f}\t'
                      f'pred_loss: {prediction_loss.item():.6f}\t'
                      f'recon_loss: {reconsruction_loss.item():.6f}\t'
                      f'kld_loss: {kld_loss.item():.6f}\t'
                      f'mmd_loss: {flags["batch_size"] * mmd_loss.item():.6f}')
            else:
                print(f'train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}'
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item() / len(data_x):.6f}\t'
                      f'recon_loss: {reconsruction_loss.item():.6f}\t'
                      f'mmd_loss: {flags["batch_size"] * mmd_loss.item():.6f}')

            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
