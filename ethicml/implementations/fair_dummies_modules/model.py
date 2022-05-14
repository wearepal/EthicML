"""FairDummies Models."""
from typing import Callable, Tuple
from typing_extensions import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ethicml.implementations.fair_dummies_modules.utility_functions import (
    DeepModel,
    DeepRegModel,
    LinearModel,
    density_estimation,
)


def covariance_diff_biased(z: torch.Tensor, w: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Covariance Difference."""
    # Center X,Xk
    m_z = z - torch.mean(z, 0, keepdim=True)
    m_w = w - torch.mean(w, 0, keepdim=True)
    # Compute covariance matrices
    szz = torch.mm(torch.t(m_z), m_z) / m_z.shape[0]
    sww = torch.mm(torch.t(w), m_w) / m_w.shape[0]

    return (szz - sww).pow(2).sum() / scale  # Compute loss


class PandasDataSet(TensorDataset):
    """Pandas based dataset."""

    def __init__(self, *dataframes: pd.DataFrame):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super().__init__(*tensors)

    def _df_to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()


# defining discriminator class (for regression)
class RegDiscriminator(nn.Module):
    """Regression Discriminator."""

    def __init__(self, inp, out=1):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 10 * inp),
            nn.ReLU(inplace=True),
            nn.Linear(10 * inp, out),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.net(x)


# defining discriminator class (for classification)
class ClassDiscriminator(nn.Module):
    """Classification Discriminator."""

    def __init__(self, out_dim: int, n_y: int, n_hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_y, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return torch.sigmoid(self.network(x))


def pretrain_adversary_fast_loader(
    dis: nn.Module,
    *,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    at: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambdas: torch.Tensor,
) -> nn.Module:
    """Pretrain adversary."""
    yhat = model(x).detach()
    dis.zero_grad()
    if len(yhat.size()) == 1:
        yhat = yhat.unsqueeze(dim=1)
    real = torch.cat((yhat, at, y), 1)
    fake = torch.cat((yhat, a, y), 1)
    in_dis = torch.cat((real, fake), 0)
    out_dis = dis(in_dis)
    labels = torch.cat((torch.ones(real.shape[0], 1), torch.zeros(fake.shape[0], 1)), 0)
    loss = (criterion(out_dis, labels) * lambdas).mean()
    loss.backward()
    optimizer.step()
    return dis


def pretrain_adversary(
    dis: nn.Module,
    *,
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambdas: torch.Tensor,
) -> nn.Module:
    """Pretrain adversary."""
    for x, y, a, at in data_loader:
        dis = pretrain_adversary_fast_loader(
            dis=dis,
            model=model,
            x=x,
            y=y,
            a=a,
            at=at,
            optimizer=optimizer,
            criterion=criterion,
            lambdas=lambdas,
        )
    return dis


def pretrain_classifier(
    model: nn.Module,
    *,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> nn.Module:
    """Pretrain classifier."""
    for x, y, _, _ in data_loader:
        model.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y.squeeze().long())
        loss.backward()
        optimizer.step()
    return model


def pretrain_regressor(
    model: nn.Module,
    *,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> nn.Module:
    """Pretrain regressor."""
    for x, y, _, _ in data_loader:
        model.zero_grad()
        yhat = model(x)
        loss = criterion(yhat.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
    return model


def pretrain_regressor_fast_loader(
    model: nn.Module,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> nn.Module:
    """Pretrain regressor."""
    model.zero_grad()
    yhat = model(x)
    loss = criterion(yhat.squeeze(), y.squeeze())
    loss.backward()
    optimizer.step()
    return model


def train_classifier(
    model: nn.Module,
    *,
    dis: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    pred_loss: nn.Module,
    dis_loss: nn.Module,
    clf_optimizer: torch.optim.Optimizer,
    adv_optimizer: torch.optim.Optimizer,
    lambdas: torch.Tensor,
    second_moment_scaling: torch.Tensor,
    dis_steps: int,
    loss_steps: int,
    num_classes: int,
) -> Tuple[nn.Module, nn.Module]:
    """Train classifier."""
    # Train adversary
    for _ in range(dis_steps):
        for x, y, a, at in data_loader:
            yhat = model(x)
            dis.zero_grad()
            if len(yhat.size()) == 1:
                yhat = yhat.unsqueeze(dim=1)
            real = torch.cat((yhat, at, y), 1)
            fake = torch.cat((yhat, a, y), 1)
            in_dis = torch.cat((real, fake), 0)
            out_dis = dis(in_dis)
            labels = torch.cat((torch.ones(real.shape[0], 1), torch.zeros(fake.shape[0], 1)), 0)
            loss_adv = (dis_loss(out_dis, labels) * lambdas).mean()
            loss_adv.backward()
            adv_optimizer.step()

    # Train predictor
    for _ in range(loss_steps):
        for x, y, a, at in data_loader:
            yhat = model(x)
            if len(yhat.size()) == 1:
                yhat = yhat.unsqueeze(dim=1)

            y_one_hot = torch.zeros(len(y), num_classes).scatter_(1, y.long(), 1.0)
            fake_one_hot = torch.cat((yhat, a, y_one_hot), 1)
            real_one_hot = torch.cat((yhat, at, y_one_hot), 1)

            loss_second_moment = covariance_diff_biased(fake_one_hot, real_one_hot)

            fake = torch.cat((yhat, a, y), 1)
            real = torch.cat((yhat, at, y), 1)

            in_dis = torch.cat((real, fake), 0)
            out_dis = dis(in_dis)
            model.zero_grad()
            out_dis = dis(in_dis)
            labels = torch.cat((torch.zeros(real.shape[0], 1), torch.ones(fake.shape[0], 1)), 0)
            loss_adv = (dis_loss(out_dis, labels) * lambdas).mean()
            clf_loss = (1.0 - lambdas) * pred_loss(yhat, y.squeeze().long())
            clf_loss += (dis_loss(dis(in_dis), labels) * lambdas).mean()
            clf_loss += lambdas * second_moment_scaling * loss_second_moment
            clf_loss.backward()
            clf_optimizer.step()

            break

    return model, dis


def inner_train_adversary_regression(
    model: nn.Module,
    *,
    dis: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    at: torch.Tensor,
    pred_loss: nn.Module,
    dis_loss: nn.Module,
    clf_optimizer: torch.optim.Optimizer,
    adv_optimizer: torch.optim.Optimizer,
    lambdas: torch.Tensor,
    second_moment_scaling: torch.Tensor,
    dis_steps: int,
    loss_steps: int,
) -> nn.Module:
    """Inner train."""
    yhat = model(x)
    dis.zero_grad()
    if len(yhat.size()) == 1:
        yhat = yhat.unsqueeze(dim=1)
    real = torch.cat((yhat, at, y), 1)
    fake = torch.cat((yhat, a, y), 1)
    in_dis = torch.cat((real, fake), 0)
    out_dis = dis(in_dis)
    labels = torch.cat((torch.ones(real.shape[0], 1), torch.zeros(fake.shape[0], 1)), 0)
    loss_adv = (dis_loss(out_dis, labels) * lambdas).mean()
    loss_adv.backward()
    adv_optimizer.step()
    return dis


def inner_train_model_regression(
    model: nn.Module,
    *,
    dis: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    at: torch.Tensor,
    pred_loss: nn.Module,
    dis_loss: nn.Module,
    clf_optimizer: torch.optim.Optimizer,
    adv_optimizer: torch.optim.Optimizer,
    lambdas: torch.Tensor,
    second_moment_scaling: torch.Tensor,
    dis_steps: int,
    loss_steps: int,
) -> nn.Module:
    """Inner train."""
    yhat = model(x)
    if len(yhat.size()) == 1:
        yhat = yhat.unsqueeze(dim=1)

    fake = torch.cat((yhat, a, y), 1)
    real = torch.cat((yhat, at, y), 1)

    loss_second_moment = covariance_diff_biased(fake, real)

    in_dis = torch.cat((real, fake), 0)
    out_dis = dis(in_dis)
    model.zero_grad()
    out_dis = dis(in_dis)
    labels = torch.cat((torch.zeros(real.shape[0], 1), torch.ones(fake.shape[0], 1)), 0)
    (dis_loss(out_dis, labels) * lambdas).mean()
    clf_loss = (1.0 - lambdas) * pred_loss(yhat.squeeze(), y.squeeze())
    clf_loss += (dis_loss(dis(in_dis), labels) * lambdas).mean()
    clf_loss += lambdas * second_moment_scaling * loss_second_moment
    clf_loss.backward()
    clf_optimizer.step()
    return model


def train_regressor_fast_loader(
    model: nn.Module,
    *,
    dis: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    at: torch.Tensor,
    pred_loss: nn.Module,
    dis_loss: nn.Module,
    clf_optimizer: torch.optim.Optimizer,
    adv_optimizer: torch.optim.Optimizer,
    lambdas: torch.Tensor,
    second_moment_scaling: torch.Tensor,
    dis_steps: int,
    loss_steps: int,
) -> Tuple[nn.Module, nn.Module]:
    """Train regressor."""
    # Train adversary
    for _ in range(dis_steps):
        dis = inner_train_adversary_regression(
            model=model,
            dis=dis,
            x=x,
            y=y,
            a=a,
            at=at,
            pred_loss=pred_loss,
            dis_loss=dis_loss,
            clf_optimizer=clf_optimizer,
            adv_optimizer=adv_optimizer,
            lambdas=lambdas,
            second_moment_scaling=second_moment_scaling,
            dis_steps=dis_steps,
            loss_steps=loss_steps,
        )

    # Train predictor
    for _ in range(loss_steps):
        model = inner_train_model_regression(
            model=model,
            dis=dis,
            x=x,
            y=y,
            a=a,
            at=at,
            pred_loss=pred_loss,
            dis_loss=dis_loss,
            clf_optimizer=clf_optimizer,
            adv_optimizer=adv_optimizer,
            lambdas=lambdas,
            second_moment_scaling=second_moment_scaling,
            dis_steps=dis_steps,
            loss_steps=loss_steps,
        )

    return model, dis


def train_regressor(
    model: nn.Module,
    *,
    dis: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    pred_loss: nn.Module,
    dis_loss: nn.Module,
    clf_optimizer: torch.optim.Optimizer,
    adv_optimizer: torch.optim.Optimizer,
    lambdas: torch.Tensor,
    second_moment_scaling: torch.Tensor,
    dis_steps: int,
    loss_steps: int,
) -> Tuple[nn.Module, nn.Module]:
    """Train the regressor."""
    # Train adversary
    for _ in range(dis_steps):
        for x, y, a, at in data_loader:
            dis = inner_train_adversary_regression(
                model=model,
                dis=dis,
                x=x,
                y=y,
                a=a,
                at=at,
                pred_loss=pred_loss,
                dis_loss=dis_loss,
                clf_optimizer=clf_optimizer,
                adv_optimizer=adv_optimizer,
                lambdas=lambdas,
                second_moment_scaling=second_moment_scaling,
                dis_steps=dis_steps,
                loss_steps=loss_steps,
            )

    # Train predictor
    for _ in range(loss_steps):
        for x, y, a, at in data_loader:
            model = inner_train_model_regression(
                model=model,
                dis=dis,
                x=x,
                y=y,
                a=a,
                at=at,
                pred_loss=pred_loss,
                dis_loss=dis_loss,
                clf_optimizer=clf_optimizer,
                adv_optimizer=adv_optimizer,
                lambdas=lambdas,
                second_moment_scaling=second_moment_scaling,
                dis_steps=dis_steps,
                loss_steps=loss_steps,
            )

    return model, dis


class EquiClassLearner:
    """Classification model."""

    def __init__(
        self,
        lr: float,
        pretrain_pred_epochs: int,
        pretrain_dis_epochs: int,
        epochs: int,
        loss_steps: int,
        dis_steps: int,
        cost_pred: nn.Module,
        in_shape: int,
        batch_size: int,
        model_type: Literal["deep_model", "linear_model"],
        lambda_vec: float,
        second_moment_scaling: float,
        num_classes: int,
    ):

        self.lr = lr
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.num_classes = num_classes

        self.model_type = model_type
        if self.model_type == "deep_model":
            self.model: nn.Module = DeepModel(in_shape=in_shape, out_shape=num_classes)
        elif self.model_type == "linear_model":
            self.model = LinearModel(in_shape=in_shape, out_shape=num_classes)
        else:
            raise NotImplementedError

        self.pred_loss = cost_pred
        self.clf_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.pretrain_pred_epochs = pretrain_pred_epochs

        self.lambdas = torch.Tensor([lambda_vec])
        self.second_moment_scaling = torch.Tensor([second_moment_scaling])

        self.dis: nn.Module = ClassDiscriminator(out_dim=1, n_y=num_classes + 1 + 1)
        self.dis_loss = nn.BCELoss(reduce=False)
        self.adv_optimizer = optim.Adam(self.dis.parameters(), lr=self.lr)

        self.pretrain_dis_epochs = pretrain_dis_epochs

        self.epochs = epochs
        self.loss_steps = loss_steps
        self.dis_steps = dis_steps

        self.scaler = StandardScaler()
        self.scale_df = lambda df, scaler: pd.DataFrame(
            scaler.transform(df), columns=df.columns, index=df.index
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit."""
        # The features are X[:,1:]
        x_train = pd.DataFrame(data=x[:, 1:])
        y_train = pd.DataFrame(data=y)
        orig_z = x[:, 0]
        z_train = pd.DataFrame(data=orig_z)
        p_success, dummy = density_estimation(y=y, a=orig_z)

        self.scaler.fit(x_train)
        x_train = x_train.pipe(self.scale_df, self.scaler)

        for _ in range(self.pretrain_pred_epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_z.shape)
            z_tilde = (random_array < p_success).astype(float)
            zt_train = pd.DataFrame(data=z_tilde)
            train_data = PandasDataSet(x_train, y_train, z_train, zt_train)
            train_loader = DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True, drop_last=True
            )

            self.model = pretrain_classifier(
                model=self.model,
                data_loader=train_loader,
                optimizer=self.clf_optimizer,
                criterion=self.pred_loss,
            )

        for _ in range(self.pretrain_dis_epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_z.shape)
            z_tilde = (random_array < p_success).astype(float)
            zt_train = pd.DataFrame(data=z_tilde)
            train_data = PandasDataSet(x_train, y_train, z_train, zt_train)
            train_loader = DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True, drop_last=True
            )

            pretrain_adversary(
                dis=self.dis,
                model=self.model,
                data_loader=train_loader,
                optimizer=self.adv_optimizer,
                criterion=self.dis_loss,
                lambdas=self.lambdas,
            )

        for _ in range(1, self.epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_z.shape)
            z_tilde = (random_array < p_success).astype(float)
            zt_train = pd.DataFrame(data=z_tilde)
            train_data = PandasDataSet(x_train, y_train, z_train, zt_train)
            train_loader = DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True, drop_last=True
            )

            self.model, self.dis = train_classifier(
                model=self.model,
                dis=self.dis,
                data_loader=train_loader,
                pred_loss=self.pred_loss,
                dis_loss=self.dis_loss,
                clf_optimizer=self.clf_optimizer,
                adv_optimizer=self.adv_optimizer,
                lambdas=self.lambdas,
                second_moment_scaling=self.second_moment_scaling,
                dis_steps=self.dis_steps,
                loss_steps=self.loss_steps,
                num_classes=self.num_classes,
            )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict."""
        x = x[:, 1:]
        x_test = pd.DataFrame(data=x)
        x_test = x_test.pipe(self.scale_df, self.scaler)

        test_data = PandasDataSet(x_test)

        with torch.no_grad():
            yhat = self.model(test_data.tensors[0])

        sm = nn.Softmax(dim=1)
        yhat = sm(yhat)
        yhat = yhat.detach().numpy()

        return yhat


class EquiRegLearner:
    """Regression model."""

    def __init__(
        self,
        lr: float,
        pretrain_pred_epochs: int,
        pretrain_dis_epochs: int,
        epochs: int,
        loss_steps: int,
        dis_steps: int,
        cost_pred: nn.Module,
        in_shape: int,
        batch_size: int,
        model_type: Literal["deep_model", "linear_model"],
        lambda_vec: float,
        second_moment_scaling: float,
        out_shape: int,
        use_standardscaler: bool = True,
    ):

        self.lr = lr
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.use_standardscaler = use_standardscaler

        self.model_type = model_type
        if self.model_type == "deep_model":
            self.model: nn.Module = DeepRegModel(in_shape=in_shape, out_shape=out_shape)
        elif self.model_type == "linear_model":
            self.model = LinearModel(in_shape=in_shape, out_shape=out_shape)
        else:
            raise NotImplementedError

        self.pred_loss = cost_pred
        self.clf_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.pretrain_pred_epochs = pretrain_pred_epochs

        self.lambdas = torch.Tensor([lambda_vec])
        self.second_moment_scaling = torch.Tensor([second_moment_scaling])

        self.dis: nn.Module = RegDiscriminator(out_shape + 1 + 1)
        self.dis_loss = nn.BCELoss(reduce=False)
        self.adv_optimizer = torch.optim.SGD(self.dis.parameters(), lr=self.lr, momentum=0.9)
        self.pretrain_dis_epochs = pretrain_dis_epochs

        self.epochs = epochs
        self.loss_steps = loss_steps
        self.dis_steps = dis_steps

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_z = StandardScaler()
        self.scaler_zt = StandardScaler()

        self.scale_df = lambda df, scaler: pd.DataFrame(
            scaler.transform(df), columns=df.columns, index=df.index
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit."""
        fast_loader = self.batch_size >= x.shape[0]
        # The features are X[:,1:]
        x_train = pd.DataFrame(data=x[:, 1:])
        y_train = pd.DataFrame(data=y)
        orig_z = x[:, 0]
        z_train = pd.DataFrame(data=orig_z)
        p_success, dummy = density_estimation(y=y, a=orig_z)

        if self.use_standardscaler:
            self.scaler_x.fit(x_train)
            x_train = x_train.pipe(self.scale_df, self.scaler_x)

            self.scaler_z.fit(z_train)
            z_train = z_train.pipe(self.scale_df, self.scaler_z)

            if self.out_shape == 1:
                self.scaler_y.fit(y_train)
                y_train = y_train.pipe(self.scale_df, self.scaler_y)

        x = torch.from_numpy(x_train.values).float()
        y = torch.from_numpy(y_train.values).float()
        a = torch.from_numpy(z_train.values).float()

        for _ in range(self.pretrain_pred_epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_z.shape)
            z_tilde = (random_array < p_success).astype(float)
            zt_train = pd.DataFrame(data=z_tilde)
            if self.use_standardscaler:
                self.scaler_zt.fit(zt_train)
                zt_train = zt_train.pipe(self.scale_df, self.scaler_zt)
            train_data = PandasDataSet(x_train, y_train, z_train, zt_train)
            train_loader = DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True, drop_last=False
            )
            self.model = (
                pretrain_regressor_fast_loader(
                    model=self.model,
                    x=x,
                    y=y,
                    optimizer=self.clf_optimizer,
                    criterion=self.pred_loss,
                )
                if fast_loader
                else pretrain_regressor(
                    model=self.model,
                    data_loader=train_loader,
                    optimizer=self.clf_optimizer,
                    criterion=self.pred_loss,
                )
            )

        for _ in range(self.pretrain_dis_epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_z.shape)
            z_tilde = (random_array < p_success).astype(float)
            zt_train = pd.DataFrame(data=z_tilde)
            if self.use_standardscaler:
                self.scaler_zt.fit(zt_train)
                zt_train = zt_train.pipe(self.scale_df, self.scaler_zt)
            train_data = PandasDataSet(x_train, y_train, z_train, zt_train)
            train_loader = DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True, drop_last=False
            )
            if fast_loader:
                pretrain_adversary_fast_loader(
                    dis=self.dis,
                    model=self.model,
                    x=x,
                    y=y,
                    a=a,
                    at=torch.from_numpy(zt_train.values).float(),
                    optimizer=self.adv_optimizer,
                    criterion=self.dis_loss,
                    lambdas=self.lambdas,
                )
            else:
                pretrain_adversary(
                    dis=self.dis,
                    model=self.model,
                    data_loader=train_loader,
                    optimizer=self.adv_optimizer,
                    criterion=self.dis_loss,
                    lambdas=self.lambdas,
                )

        for _ in range(1, self.epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_z.shape)
            z_tilde = (random_array < p_success).astype(float)
            zt_train = pd.DataFrame(data=z_tilde)
            if self.use_standardscaler:
                self.scaler_zt.fit(zt_train)
                zt_train = zt_train.pipe(self.scale_df, self.scaler_zt)
            train_data = PandasDataSet(x_train, y_train, z_train, zt_train)
            train_loader = DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True, drop_last=False
            )
            self.model, self.dis = (
                train_regressor_fast_loader(
                    model=self.model,
                    dis=self.dis,
                    x=x,
                    y=y,
                    a=a,
                    at=torch.from_numpy(zt_train.values).float(),
                    pred_loss=self.pred_loss,
                    dis_loss=self.dis_loss,
                    clf_optimizer=self.clf_optimizer,
                    adv_optimizer=self.adv_optimizer,
                    lambdas=self.lambdas,
                    second_moment_scaling=self.second_moment_scaling,
                    dis_steps=self.dis_steps,
                    loss_steps=self.loss_steps,
                )
                if fast_loader
                else train_regressor(
                    model=self.model,
                    dis=self.dis,
                    data_loader=train_loader,
                    pred_loss=self.pred_loss,
                    dis_loss=self.dis_loss,
                    clf_optimizer=self.clf_optimizer,
                    adv_optimizer=self.adv_optimizer,
                    lambdas=self.lambdas,
                    second_moment_scaling=self.second_moment_scaling,
                    dis_steps=self.dis_steps,
                    loss_steps=self.loss_steps,
                )
            )

    def predict(self, x: np.ndarray) -> float:
        """Predict."""
        x = x[:, 1:]
        x_test = pd.DataFrame(data=x)

        if self.use_standardscaler:
            x_test = x_test.pipe(self.scale_df, self.scaler_x)

        test_data = PandasDataSet(x_test)

        with torch.no_grad():
            yhat = self.model(test_data.tensors[0]).squeeze().detach().numpy()

        if self.out_shape == 1 and self.use_standardscaler:
            out = self.scaler_y.inverse_transform(yhat.reshape(-1, 1)).squeeze()
        elif self.out_shape == 1:
            out = yhat.squeeze()
        else:
            out = 0 * yhat
            out[:, 0] = np.min(yhat, axis=1)
            out[:, 1] = np.max(yhat, axis=1)

        return out
