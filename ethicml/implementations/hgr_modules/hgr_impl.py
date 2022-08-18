"""HGR."""
# Code follows the implemention provided in
# https://github.com/criteo-research/continuous-fairness
# The function for measuring HGR is in the facl package, can be downloaded from
# https://github.com/criteo-research/continuous-fairness/tree/master/facl/independence
from __future__ import annotations
import random
from typing_extensions import Self

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from ethicml.implementations.pytorch_common import (
    DeepModel,
    DeepRegModel,
    LinearModel,
    make_dataset_and_loader,
)
from ethicml.utility.data_structures import DataTuple, ModelType

from .density_estimation import Kde
from .facl_hgr import chi_2_cond


def chi_squared_l1_kde(x, y, z):
    """Chi Squared."""
    return torch.mean(chi_2_cond(x, y, z, Kde))


class HgrRegLearner:
    """HGR Regression."""

    def __init__(self, lr, epochs, mu, cost_pred, in_shape, out_shape, batch_size, model_type):

        self.in_shape = in_shape
        self.model_type = model_type

        # Data normalization
        self.x_scaler = StandardScaler()
        self.a_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # EO penalty
        self.mu = mu

        # Loss optimization
        self.cost_pred = cost_pred
        self.epochs = epochs
        self.lr_loss = lr
        self.batch_size = batch_size

        self.out_shape = out_shape
        if self.model_type is ModelType.deep:
            self.model: nn.Module = DeepRegModel(in_shape=in_shape, out_shape=out_shape)
        elif self.model_type is ModelType.linear:
            self.model = LinearModel(in_shape=in_shape, out_shape=out_shape)

        else:
            raise NotImplementedError
        self.loss_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_loss)

    def internal_epoch(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """Internal epochs."""
        # fit pred func
        epoch_losses = []
        for x, s, y in dataloader:
            self.loss_optimizer.zero_grad()

            batch_yhat = self.model(x)

            # utility loss
            pred_loss = self.cost_pred(batch_yhat, y)

            if self.out_shape == 1:
                dis_loss = chi_squared_l1_kde(batch_yhat, s, y)
            else:
                dis_loss = sum(
                    chi_squared_l1_kde(batch_yhat[:, out_id], s, y)
                    for out_id in range(batch_yhat.shape[1])
                )

            # combine utility + 'distance to equalized odds'
            loss = (1 - self.mu) * pred_loss + self.mu * dis_loss

            loss.backward()
            self.loss_optimizer.step()

            epoch_losses.append(loss.detach().cpu().numpy())

        return np.mean(epoch_losses)

    def run_epochs(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Run Epochs."""
        for _ in range(self.epochs):
            self.internal_epoch(dataloader)

    def fit(self, train: DataTuple, seed: int) -> None:
        """Fit."""
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        train_data, train_loader = make_dataset_and_loader(
            train, batch_size=self.batch_size, shuffle=True, seed=seed, drop_last=True
        )
        self.run_epochs(train_loader)

    @torch.no_grad()
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict."""
        xp = torch.from_numpy(self.x_scaler.transform(x.to_numpy())).float()
        yhat = self.model(xp).detach()
        yhat = yhat.detach().cpu().numpy()

        if self.out_shape == 1:
            out = self.y_scaler.inverse_transform(yhat.reshape(-1, 1)).squeeze()
        else:
            out = 0 * yhat
            out[:, 0] = np.min(yhat, axis=1)
            out[:, 1] = np.max(yhat, axis=1)

        return out


class HgrClassLearner:
    """HGR Class."""

    def __init__(
        self,
        lr: float,
        epochs: int,
        mu: float,
        cost_pred: nn.Module,
        in_shape: int,
        out_shape: int,
        batch_size: int,
        model_type: ModelType,
    ):

        self.in_shape = in_shape
        self.num_classes = out_shape

        # EO penalty
        self.mu = mu

        # Loss optimization
        self.cost_pred = cost_pred
        self.epochs = epochs
        self.lr_loss = lr
        self.batch_size = batch_size

        self.model_type = model_type
        if self.model_type is ModelType.deep:
            self.model: nn.Module = DeepModel(in_shape=in_shape, out_shape=out_shape)
        elif self.model_type is ModelType.linear:
            self.model = LinearModel(in_shape=in_shape, out_shape=out_shape)
        else:
            raise NotImplementedError

        self.loss_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_loss)

    def internal_epoch(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Internal Epoch."""
        # fit pred func
        epoch_losses = []
        for x, s, y in dataloader:
            self.loss_optimizer.zero_grad()
            batch_yhat = self.model(x)

            # utility loss
            pred_loss = self.cost_pred(batch_yhat, y.long())

            dis_loss = sum(
                chi_squared_l1_kde(batch_yhat[:, out_id], s, y.float())
                for out_id in range(batch_yhat.shape[1])
            )

            # combine utility + 'distance to equalized odds'
            loss = (1 - self.mu) * pred_loss + self.mu * dis_loss

            loss.backward()
            self.loss_optimizer.step()

            epoch_losses.append(loss.cpu().detach().numpy())

    def run_epochs(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Run epochs."""
        for _ in range(self.epochs):
            self.internal_epoch(dataloader)

    def fit(self, train: DataTuple, seed: int) -> Self:  # type: ignore[valid-type]
        """Fit."""
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        # train
        train_data, train_loader = make_dataset_and_loader(
            train, batch_size=self.batch_size, shuffle=True, seed=seed, drop_last=True
        )
        self.model.train()
        self.run_epochs(train_loader)
        return self

    @torch.no_grad()
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict."""
        xp = torch.from_numpy(x.to_numpy()).float()
        self.model.eval()
        yhat = self.model(xp)
        sm = nn.Softmax(dim=1)
        yhat = sm(yhat)
        return yhat.detach().numpy()
