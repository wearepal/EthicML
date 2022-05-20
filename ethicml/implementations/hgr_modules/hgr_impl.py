"""HGR."""
# Code follows the implemention provided in
# https://github.com/criteo-research/continuous-fairness
# The function for measuring HGR is in the facl package, can be downloaded from
# https://github.com/criteo-research/continuous-fairness/tree/master/facl/independence

import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..pytorch_common import DeepModel, DeepRegModel, LinearModel
from .density_estimation import Kde
from .facl_hgr import chi_2_cond
from .utility import compute_acc


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
        if self.model_type == "deep_model":
            self.model = DeepRegModel(in_shape=in_shape, out_shape=out_shape)
        elif self.model_type == "linear_model":
            self.model = LinearModel(in_shape=in_shape, out_shape=out_shape)
        else:
            raise NotImplementedError
        self.loss_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_loss)

    def internal_epoch(self, x_, a_, y_):
        """Internal epochs."""
        # shuffle data
        shuffle_idx = np.arange(x_.shape[0])
        np.random.shuffle(shuffle_idx)
        X = x_.clone()[shuffle_idx]
        A = a_.clone()[shuffle_idx]
        Y = y_.clone()[shuffle_idx]

        # fit pred func
        batch_size = self.batch_size
        epoch_losses = []
        for idx in range(0, X.shape[0], batch_size):
            self.loss_optimizer.zero_grad()

            batch_x = X[idx : min(idx + batch_size, X.shape[0]), :]
            batch_a = A[idx : min(idx + batch_size, X.shape[0])]
            batch_y = Y[idx : min(idx + batch_size, Y.shape[0])]
            batch_yhat = self.model(batch_x)

            # utility loss
            pred_loss = self.cost_pred(batch_yhat, batch_y)

            if self.out_shape == 1:
                dis_loss = chi_squared_l1_kde(batch_yhat, batch_a, batch_y)
            else:
                dis_loss = 0
                for out_id in range(batch_yhat.shape[1]):
                    dis_loss += chi_squared_l1_kde(batch_yhat[:, out_id], batch_a, batch_y)

            # combine utility + 'distance to equalized odds'
            loss = (1 - self.mu) * pred_loss + self.mu * dis_loss

            loss.backward()
            self.loss_optimizer.step()

            epoch_losses.append(loss.detach().cpu().numpy())

        epoch_loss = np.mean(epoch_losses)

        return epoch_loss

    def run_epochs(self, X, A, Y):
        """Run Epochs."""
        for epoch in range(self.epochs):
            self.internal_epoch(X, A, Y)

    def fit(self, X, Y):
        """Fit."""
        # The features are in X[:,1:]
        A = X[:, 0]
        self.input_A = A
        X = X[:, 1:]

        self.x_scaler.fit(X)
        self.a_scaler.fit(A.reshape(-1, 1))

        # normalize y if classic regression
        if self.out_shape == 1:
            self.y_scaler.fit(Y.reshape(-1, 1))
            Yp = torch.from_numpy(self.y_scaler.transform(Y.reshape(-1, 1)).squeeze()).float()
        else:
            Yp = torch.from_numpy(Y).float()

        Xp = torch.from_numpy(self.x_scaler.transform(X)).float()
        Ap = torch.from_numpy(self.a_scaler.transform(A.reshape(-1, 1)).squeeze()).float()

        # evaluate at init
        Yhat = self.model(Xp)
        print('Init : Cost Pred = ' + str(self.cost_pred(Yhat, Yp).detach().cpu().numpy()))
        sys.stdout.flush()

        # train
        self.run_epochs(Xp, Ap, Yp)

        # evaluate
        Yhat = self.model(Xp)
        print(
            'mu = '
            + str(self.mu)
            + ' Cost Pred = '
            + str(self.cost_pred(Yhat, Yp).detach().cpu().numpy())
        )
        sys.stdout.flush()

    def predict(self, X):
        """Predict."""
        X = X[:, 1:]
        Xp = torch.from_numpy(self.x_scaler.transform(X)).float()
        Yhat = self.model(Xp).detach()
        Yhat = Yhat.detach().cpu().numpy()

        if self.out_shape == 1:
            out = self.y_scaler.inverse_transform(Yhat.reshape(-1, 1)).squeeze()
        else:
            out = 0 * Yhat
            out[:, 0] = np.min(Yhat, axis=1)
            out[:, 1] = np.max(Yhat, axis=1)

        return out


class HgrClassLearner:
    """HGR Class."""

    def __init__(self, lr, epochs, mu, cost_pred, in_shape, out_shape, batch_size, model_type):

        self.in_shape = in_shape
        self.num_classes = out_shape

        # Data normalization
        self.X_scaler = StandardScaler()
        self.A_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()

        # EO penalty
        self.mu = mu

        # Loss optimization
        self.cost_pred = cost_pred
        self.epochs = epochs
        self.lr_loss = lr
        self.batch_size = batch_size

        self.model_type = model_type
        if self.model_type == "deep_model":
            self.model = DeepModel(in_shape=in_shape, out_shape=out_shape)
        elif self.model_type == "linear_model":
            self.model = LinearModel(in_shape=in_shape, out_shape=out_shape)
        else:
            raise

        self.loss_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_loss)

    def internal_epoch(self, X_, A_, Y_, Yc_):
        """Internal Epoch."""
        # shuffle data
        shuffle_idx = np.arange(X_.shape[0])
        np.random.shuffle(shuffle_idx)
        X = X_.clone()[shuffle_idx]
        A = A_.clone()[shuffle_idx]
        Y = Y_.clone()[shuffle_idx]
        Yc = Yc_.clone()[shuffle_idx]

        # fit pred func
        batch_size = self.batch_size
        epoch_losses = []
        for idx in range(0, X.shape[0], batch_size):
            self.loss_optimizer.zero_grad()

            batch_x = X[idx : min(idx + batch_size, X.shape[0]), :]
            batch_a = A[idx : min(idx + batch_size, X.shape[0])]
            batch_yc = Yc[idx : min(idx + batch_size, Yc.shape[0])]
            batch_y = Y[idx : min(idx + batch_size, Y.shape[0])]

            batch_Yhat = self.model(batch_x)

            # utility loss
            pred_loss = self.cost_pred(batch_Yhat, batch_yc)

            dis_loss = sum(
                chi_squared_l1_kde(batch_Yhat[:, out_id], batch_a, batch_yc.float())
                for out_id in range(batch_Yhat.shape[1])
            )

            # combine utility + 'distance to equalized odds'
            loss = (1 - self.mu) * pred_loss + self.mu * dis_loss

            loss.backward()
            self.loss_optimizer.step()

            epoch_losses.append(loss.cpu().detach().numpy())

        return np.mean(epoch_losses)

    def run_epochs(self, X, A, Y, Yc):
        """Run epochs."""
        for _ in range(self.epochs):
            self.internal_epoch(X, A, Y, Yc)

    def fit(self, X, Y_categorial):
        """Fit."""
        # The features are X[:,1:]
        A = X[:, 0]
        self.input_A = A
        X = X[:, 1:]

        self.X_scaler.fit(X)

        Xp = torch.from_numpy(self.X_scaler.transform(X)).float()
        Yc = torch.from_numpy(Y_categorial).long()
        Yp = torch.zeros(len(Yc), self.num_classes).scatter_(1, Yc.unsqueeze(1), 1.0).long()

        self.A_scaler.fit(A.reshape(-1, 1))
        Ap = torch.from_numpy(self.A_scaler.transform(A.reshape(-1, 1)).squeeze()).float()

        # evaluate at init
        Yhat = self.model(Xp)

        print(
            'Init : Loss = '
            + str(self.cost_pred(Yhat, Yc).detach().numpy())
            + ' ACC = '
            + str(compute_acc(Yhat, Yc))
        )

        # train
        self.run_epochs(Xp, Ap, Yp, Yc)

        # evaluate
        Yhat = self.model(Xp)
        print(
            'mu = '
            + str(self.mu)
            + ' Loss = '
            + str(self.cost_pred(Yhat, Yc).detach().numpy())
            + ' ACC = '
            + str(compute_acc(Yhat, Yc))
        )
        return self

    def predict(self, X):
        """Predict."""
        X = X[:, 1:]

        Xp = torch.from_numpy(self.X_scaler.transform(X)).float()
        Yhat = self.model(Xp)
        sm = nn.Softmax(dim=1)
        Yhat = sm(Yhat)
        Yhat = Yhat.detach().numpy()

        return Yhat
