"""Code is taken from https://github.com/equialgo/fairness-in-ml.

Original implementation is modified to handle regression and multi-class
classification problems
"""
from __future__ import annotations
from typing import Literal
from typing_extensions import Self

import numpy as np
import pandas as pd
from ranzen import implements
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ethicml import DataTuple
from ethicml.implementations.pytorch_common import (
    DeepModel,
    DeepRegModel,
    LinearModel,
    make_dataset_and_loader,
)
from ethicml.utility.data_structures import ModelType


class Adversary(nn.Module):
    """Adversarial model."""

    def __init__(self, n_sensitive: int, n_y: int, n_hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_y, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    @implements(nn.Module)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))


def pretrain_adversary(
    adv: nn.Module,
    clf: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    lambdas: torch.Tensor,
) -> nn.Module:
    """Pretrain adversary."""
    for x, z, y in data_loader:
        p_y = clf(x).detach()
        adv.zero_grad()
        if len(p_y.size()) == 1:
            p_y = p_y.unsqueeze(dim=1)
        y_ = F.one_hot(y.long())
        in_adv = torch.cat((p_y, y_), 1)
        p_z = adv(in_adv)
        loss = (criterion(p_z, z.unsqueeze(1)) * lambdas).mean()
        loss.backward()
        optimizer.step()
    return adv


def train_loop(
    clf: nn.Module,
    adv: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    clf_criterion: nn.Module,
    adv_criterion: nn.Module,
    clf_optimizer: torch.optim.Optimizer,
    adv_optimizer: torch.optim.Optimizer,
    lambdas: torch.Tensor,
) -> tuple[nn.Module, nn.Module]:
    """Train model."""
    # Train adversary
    for x, z, y in data_loader:
        p_y = clf(x)
        if len(p_y.size()) == 1:
            p_y = p_y.unsqueeze(dim=1)
        adv.zero_grad()
        y_ = F.one_hot(y.long())
        in_adv = torch.cat((p_y, y_), 1)
        p_z = adv(in_adv)
        loss_adv = (adv_criterion(p_z, z.unsqueeze(1)) * lambdas).mean()
        loss_adv.backward()
        adv_optimizer.step()

    # Train predictor on single batch
    for x, z, y in data_loader:
        p_y = clf(x)
        if len(p_y.size()) == 1:
            p_y = p_y.unsqueeze(dim=1)
        y_ = F.one_hot(y.long())
        in_adv = torch.cat((p_y, y_), 1)
        p_z = adv(in_adv)  # TODO: This is unused?
        clf.zero_grad()
        p_z = adv(in_adv)
        loss_adv = (adv_criterion(p_z, z.unsqueeze(1)) * lambdas).mean()  # TODO: This is unused?
        clf_loss = (1.0 - lambdas) * clf_criterion(p_y, y.squeeze().long()) - (
            adv_criterion(adv(in_adv), z.unsqueeze(1)) * lambdas
        ).mean()
        clf_loss.backward()
        clf_optimizer.step()
        break

    return clf, adv


def train_regressor(
    clf: nn.Module,
    adv: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    clf_criterion: nn.Module,
    adv_criterion: nn.Module,
    clf_optimizer: torch.optim.Optimizer,
    adv_optimizer: torch.optim.Optimizer,
    lambdas: torch.Tensor,
) -> tuple[nn.Module, nn.Module]:
    """Train regression model."""
    # Train adversary
    for x, z, y in data_loader:
        p_y = clf(x)
        if len(p_y.size()) == 1:
            p_y = p_y.unsqueeze(dim=1)
        adv.zero_grad()
        in_adv = torch.cat((p_y, y), 1)
        p_z = adv(in_adv)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        loss_adv.backward()
        adv_optimizer.step()

    # Train predictor on single batch
    for x, z, y in data_loader:
        p_y = clf(x)
        if len(p_y.size()) == 1:
            p_y = p_y.unsqueeze(dim=1)
        in_adv = torch.cat((p_y, y), 1)
        p_z = adv(in_adv)  # TODO: This is unused?
        clf.zero_grad()
        p_z = adv(in_adv)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()  # TODO: This is unused?
        clf_loss = (1.0 - lambdas) * clf_criterion(p_y.squeeze(), y.squeeze()) - (
            adv_criterion(adv(in_adv), z) * lambdas
        ).mean()
        clf_loss.backward()
        clf_optimizer.step()
        break

    return clf, adv


def pretrain_classifier(
    clf: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> nn.Module:
    """Pretrain classification model."""
    for x, _, y in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        loss = criterion(p_y, y.squeeze().long())
        loss.backward()
        optimizer.step()
    return clf


def pretrain_regressor(
    clf: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> nn.Module:
    """Pretrain regression model."""
    for x, _, y in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        loss = criterion(p_y.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
    return clf


class AdvDebiasingClassLearner:
    """Adversarial Debiasing classifier."""

    def __init__(
        self,
        lr: float,
        n_clf_epochs: int,
        n_adv_epochs: int,
        n_epoch_combined: int,
        cost_pred: nn.Module,
        in_shape: int,
        batch_size: int,
        model_type: ModelType,
        num_classes: int,
        lambda_vec: float,
    ):

        self.lr = lr
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.num_classes = num_classes

        self.model_type = model_type
        if self.model_type is ModelType.deep:
            self.clf: nn.Module = DeepModel(in_shape=in_shape, out_shape=num_classes)
        elif self.model_type is ModelType.linear:
            self.clf = LinearModel(in_shape=in_shape, out_shape=num_classes)
        else:
            raise NotImplementedError

        self.clf_criterion = cost_pred
        self.clf_optimizer = optim.Adam(self.clf.parameters(), lr=self.lr)

        self.n_clf_epochs = n_clf_epochs

        self.lambdas = torch.Tensor([lambda_vec])

        self.adv: nn.Module = Adversary(n_sensitive=1, n_y=num_classes * 2)
        self.adv_criterion = nn.BCELoss(reduction='none')
        self.adv_optimizer = optim.Adam(self.adv.parameters(), lr=self.lr)

        self.n_adv_epochs = n_adv_epochs

        self.n_epoch_combined = n_epoch_combined

    def fit(self, train: DataTuple, seed: int) -> Self:  # type: ignore[valid-type]
        """Fit."""
        train_data, train_loader = make_dataset_and_loader(
            train, batch_size=self.batch_size, shuffle=True, seed=seed, drop_last=True
        )

        for _ in range(self.n_clf_epochs):
            self.clf = pretrain_classifier(
                self.clf, train_loader, self.clf_optimizer, self.clf_criterion
            )

        for _ in range(self.n_adv_epochs):
            pretrain_adversary(
                self.adv,
                self.clf,
                train_loader,
                self.adv_optimizer,
                self.adv_criterion,
                self.lambdas,
            )

        for _ in range(1, self.n_epoch_combined):
            self.clf, self.adv = train_loop(
                self.clf,
                self.adv,
                train_loader,
                self.clf_criterion,
                self.adv_criterion,
                self.clf_optimizer,
                self.adv_optimizer,
                self.lambdas,
            )
        return self

    @torch.no_grad()
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict."""
        x = torch.from_numpy(x.to_numpy()).float()
        self.clf.eval()
        yhat = self.clf(x)
        sm = nn.Softmax(dim=1)
        yhat = sm(yhat)
        yhat = yhat.detach().numpy()

        return yhat


class AdvDebiasingRegLearner:
    """Adversarial Debiasing Learner."""

    def __init__(
        self,
        lr: float,
        n_clf_epochs: int,
        n_adv_epochs: int,
        n_epoch_combined: int,
        cost_pred: nn.Module,
        in_shape: int,
        batch_size: int,
        model_type: ModelType,
        out_shape: int,
        lambda_vec: float,
    ):

        self.lr = lr
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.model_type = model_type
        if self.model_type is ModelType.deep:
            self.clf: nn.Module = DeepRegModel(in_shape=in_shape, out_shape=out_shape)
        elif self.model_type is ModelType.linear:
            self.clf = LinearModel(in_shape=in_shape, out_shape=out_shape)
        else:
            raise NotImplementedError

        self.clf_criterion = cost_pred
        self.clf_optimizer = optim.Adam(self.clf.parameters(), lr=self.lr)

        self.N_CLF_EPOCHS = n_clf_epochs

        self.lambdas = torch.Tensor([lambda_vec])

        self.adv: nn.Module = Adversary(n_sensitive=1, n_y=out_shape + 1)
        self.adv_criterion = nn.BCELoss(reduce=False)
        self.adv_optimizer = optim.Adam(self.adv.parameters(), lr=self.lr)

        self.n_adv_epochs = n_adv_epochs

        self.n_epoch_combined = n_epoch_combined

    def fit(self, train: DataTuple, seed: int) -> Self:  # type: ignore[valid-type]
        """Fit."""
        # The features are X[:,1:]

        _, train_loader = make_dataset_and_loader(
            train, batch_size=self.batch_size, shuffle=True, seed=seed, drop_last=True
        )

        for _ in range(self.N_CLF_EPOCHS):
            self.clf = pretrain_regressor(
                self.clf, train_loader, self.clf_optimizer, self.clf_criterion
            )

        for _ in range(self.n_adv_epochs):
            pretrain_adversary(
                self.adv,
                self.clf,
                train_loader,
                self.adv_optimizer,
                self.adv_criterion,
                self.lambdas,
            )

        for _ in range(1, self.n_epoch_combined):
            self.clf, self.adv = train_regressor(
                self.clf,
                self.adv,
                train_loader,
                self.clf_criterion,
                self.adv_criterion,
                self.clf_optimizer,
                self.adv_optimizer,
                self.lambdas,
            )
        return self

    @torch.no_grad()
    def predict(self, x: pd.DataFrame) -> torch.Tensor:
        """Predict."""
        x = torch.from_numpy(x.to_numpy()).float()
        self.clf.eval()
        yhat = self.clf(x).squeeze().detach().numpy()
        if self.out_shape == 1:
            out = yhat
        else:
            out = 0 * yhat
            out[:, 0] = np.min(yhat, axis=1)
            out[:, 1] = np.max(yhat, axis=1)
        return out
