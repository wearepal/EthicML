"""Fair Dummies utility functions."""
from typing import List, Optional, Tuple, Union
from typing_extensions import Literal

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


def density_estimation(
    y: np.ndarray, *, a: np.ndarray, y_test: Optional[np.ndarray] = None
) -> Tuple[List[float], List[float]]:
    """Estimate the distribusion of P{A|Y}."""
    if y_test is None:
        y_test = []
    assert y_test is not None
    bandwidth = np.sqrt(max(np.median(np.abs(y)), 0.01))

    kde_0 = KernelDensity(kernel='linear', bandwidth=bandwidth).fit(y[a == 0][:, np.newaxis])
    kde_1 = KernelDensity(kernel='linear', bandwidth=bandwidth).fit(y[a == 1][:, np.newaxis])

    log_dens_0 = np.exp(np.squeeze(kde_0.score_samples(y[:, np.newaxis])))
    log_dens_1 = np.exp(np.squeeze(kde_1.score_samples(y[:, np.newaxis])))
    p_0 = np.sum(a == 0) / a.shape[0]
    p_1 = 1 - p_0

    # p(A=1|y) = p(y|A=1)p(A=1) / (p(y|A=1)p(A=1) + p(y|A=0)p(A=0))
    p_success = (log_dens_1 * p_1) / (log_dens_1 * p_1 + log_dens_0 * p_0 + 1e-10)

    p_success_test = []
    if len(y_test) > 0:
        log_dens_0_test = np.exp(np.squeeze(kde_0.score_samples(y_test[:, np.newaxis])))
        log_dens_1_test = np.exp(np.squeeze(kde_1.score_samples(y_test[:, np.newaxis])))
        p_success_test = (log_dens_1_test * p_1) / (
            log_dens_1_test * p_1 + log_dens_0_test * p_0 + 1e-10
        )

    return p_success, p_success_test


class LinearModel(torch.nn.Module):
    """Define linear model."""

    def __init__(self, in_shape: int = 1, out_shape: int = 2):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.build_model()

    def build_model(self) -> None:
        """Build Model."""
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.out_shape, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return torch.squeeze(self.base_model(x))


class DeepModel(torch.nn.Module):
    """Define deep neural net model for classification."""

    def __init__(self, in_shape: int = 1, out_shape: int = 1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64
        self.dropout = 0.5
        self.out_shape = out_shape
        self.build_model()

    def build_model(self) -> None:
        """Build Model."""
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.dim_h, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_h, self.out_shape, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return torch.squeeze(self.base_model(x))


class DeepRegModel(torch.nn.Module):
    """Define deep model for regression."""

    def __init__(self, in_shape: int = 1, out_shape: int = 1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64  # in_shape*10
        self.out_shape = out_shape
        self.build_model()

    def build_model(self) -> None:
        """Build model."""
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.dim_h, bias=True),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.out_shape, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return torch.squeeze(self.base_model(x))


class DeepProbaModel(torch.nn.Module):
    """Define deep regression model, used by the fair dummies test."""

    def __init__(self, in_shape: int = 1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64  # in_shape*10
        self.dropout = 0.5
        self.out_shape = 1
        self.build_model()

    def build_model(self) -> None:
        """Build Model."""
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.dim_h, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_h, self.out_shape, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return torch.squeeze(self.base_model(x))


def calc_accuracy(
    outputs: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:  # Care outputs are going to be in dimension 2
    """Accuracy."""
    max_vals, max_indices = torch.max(outputs, 1)
    return (max_indices == y).sum().detach().cpu().numpy() / max_indices.size()[0]


def compute_acc(yhat: torch.Tensor, y: torch.Tensor) -> float:
    """Accuracy."""
    _, predicted = torch.max(yhat, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return correct / total


def compute_acc_numpy(yhat: np.ndarray, y: np.ndarray) -> float:
    """Accuracy."""
    yhat = torch.from_numpy(yhat)
    y = torch.from_numpy(y)

    return compute_acc(yhat, y)


def pytorch_standard_scaler(x: torch.Tensor) -> torch.Tensor:
    """Std Scaler, torch version."""
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x


# fit a neural netwok on a given data, used by the fair dummies test
class GeneralLearner:
    """General Learner."""

    def __init__(
        self,
        lr: float,
        epochs: int,
        cost_func: nn.Module,
        in_shape: int,
        batch_size: int,
        model_type: Literal["deep_proba", "deep_regression"],
        out_shape: int = 1,
    ):

        # input dim
        self.in_shape = in_shape

        # output dim
        self.out_shape = out_shape

        # Data normalization
        self.x_scaler = StandardScaler()

        # learning rate
        self.lr = lr

        # number of epochs
        self.epochs = epochs

        # cost to minimize
        self.cost_func = cost_func

        # define a predictive model
        self.model_type = model_type
        if self.model_type == "deep_proba":
            self.model: nn.Module = DeepProbaModel(in_shape=in_shape)
        elif self.model_type == "deep_regression":
            self.model = DeepModel(in_shape=in_shape, out_shape=self.out_shape)
        else:
            raise NotImplementedError

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        # minibatch size
        self.batch_size = batch_size

    def internal_epoch(self, x_: torch.Tensor, y_: torch.Tensor) -> np.ndarray:
        """Fit a model by sweeping over all data points."""
        # shuffle data
        shuffle_idx = np.arange(x_.shape[0])
        np.random.shuffle(shuffle_idx)
        X = x_.clone()[shuffle_idx]
        Y = y_.clone()[shuffle_idx]

        # fit pred func
        self.model.train()

        batch_size = self.batch_size
        epoch_losses = []

        for idx in range(0, X.shape[0], batch_size):
            self.optimizer.zero_grad()

            batch_x = X[idx : min(idx + batch_size, X.shape[0]), :]
            batch_y = Y[idx : min(idx + batch_size, Y.shape[0])]

            # utility loss
            batch_yhat = self.model(batch_x)
            loss = self.cost_func(batch_yhat, batch_y)

            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.cpu().detach().numpy())

        return np.mean(epoch_losses)

    def run_epochs(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Run epochs."""
        for _ in range(self.epochs):
            self.internal_epoch(x, y)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit a model on training data."""
        self.x_scaler.fit(x)

        xp = torch.from_numpy(self.x_scaler.transform(x)).float()
        yp = torch.from_numpy(y).float()

        # evaluate at init
        self.model.eval()
        yhat = self.model(xp)

        print(f'Init Loss = {str(self.cost_func(yhat, yp).detach().numpy())}')

        self.model.train()
        self.run_epochs(xp, yp)

        # evaluate
        self.model.eval()
        yhat = self.model(xp)

        print(f'Final Loss = {str(self.cost_func(yhat, yp).detach().numpy())}')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict output."""
        self.model.eval()

        xp = torch.from_numpy(self.x_scaler.transform(x)).float()
        yhat = self.model(xp)
        yhat = yhat.detach().numpy()

        return yhat


def fair_dummies_test_regression(
    yhat_cal: torch.Tensor,
    *,
    a_cal: np.ndarray,
    y_cal: np.ndarray,
    yhat: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    num_reps: int = 1,
    num_p_val_rep: int = 1000,
    reg_func_name: str = "Net",
    lr: float = 0.1,
    return_vec: bool = False,
) -> Union[float, List[float]]:
    """Run the fair dummies test.

    Yhat_cal, A_cal, Y_cal: are used to fit a model that formulates the test statistics
    Yhat, A, Y: variables in which we test whether Yhat is indpendent on A given Y.
    """
    p_success, dummy = density_estimation(
        y=np.concatenate((y_cal, y), 0), a=np.concatenate((a_cal, a), 0)
    )
    p_success = p_success[y_cal.shape[0] :]

    out_shape = yhat.shape[1] if len(yhat.shape) > 1 else 1
    y_cal = y_cal[:, np.newaxis]
    y = y[:, np.newaxis]

    test_i = []
    for _ in range(num_reps):
        # fit regressor
        if reg_func_name == "Net":
            regr = GeneralLearner(
                lr=lr,
                epochs=200,
                cost_func=nn.MSELoss(),
                in_shape=2,
                batch_size=128,
                model_type="deep_regression",
                out_shape=out_shape,
            )

        elif reg_func_name == "RF":
            regr = RandomForestRegressor(n_estimators=10)
        features_cal = np.concatenate((a_cal[:, np.newaxis], y_cal), 1)
        regr.fit(features_cal, yhat_cal)

        # compute error on holdout points
        features_orig = np.concatenate((a[:, np.newaxis], y), 1)
        output_orig = regr.predict(features_orig)
        est_orig_err = np.mean((yhat - output_orig) ** 2)

        # generate A and compare
        est_fake_err = np.zeros(num_p_val_rep)
        for inter_p_value in range(num_p_val_rep):
            random_array = np.random.uniform(low=0.0, high=1.0, size=a.shape)
            a_tilde = (random_array < p_success).astype(float)

            features_fake = np.concatenate((a_tilde[:, np.newaxis], y), 1)
            output_fake = regr.predict(features_fake)
            est_fake_err[inter_p_value] = np.mean((yhat - output_fake) ** 2)

        p_val = 1.0 / (num_p_val_rep + 1) * (1 + sum(est_orig_err >= est_fake_err))

        test_i.append(p_val)

    print(
        "Fair dummies test (regression score), p-value:", np.mean(test_i)
    )  # should be uniform under ind.

    out: Union[float, List[float]] = test_i[0]
    if return_vec:
        out = test_i

    return out


def classification_score(*, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Classification score."""
    assert y <= len(y_hat)
    return y_hat[int(y)]


def fair_dummies_test_classification(
    yhat_cal: np.ndarray,
    *,
    a_cal: np.ndarray,
    y_cal: np.ndarray,
    yhat: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    num_reps: int = 10,
    num_p_val_rep: int = 1000,
    reg_func_name: str = "Net",
) -> float:
    """Run the fair dummies test.

    Yhat_cal, A_cal, Y_cal: are used to fit a model that formulates the test statistics
    Yhat, A, Y: variables in which we test whether Yhat is indpendent on A given Y
    """
    p_success, dummy = density_estimation(
        y=np.concatenate((y_cal, y), 0), a=np.concatenate((a_cal, a), 0)
    )
    p_success = p_success[y_cal.shape[0] :]

    yhat_cal_score = np.array(
        [classification_score(y_hat=yhat_cal[i], y=y_cal[i]) for i in range(yhat_cal.shape[0])],
        dtype=float,
    )
    yhat_score = np.array(
        [classification_score(y_hat=yhat[i], y=y[i]) for i in range(y.shape[0])], dtype=float
    )

    def get_dummies(labels):
        num_datapoints = len(labels)
        row_ind = np.arange(num_datapoints)
        return csr_matrix((np.ones(num_datapoints), (row_ind, labels)), dtype=float).todense()

    y_cal = get_dummies(y_cal)
    y = get_dummies(y)

    test_i = []
    err_func = nn.BCELoss()
    for i in range(num_reps):

        features_cal = np.concatenate((a_cal[:, np.newaxis], y_cal), 1)

        # fit regressor
        if reg_func_name == "RF":
            regr = RandomForestRegressor(n_estimators=10)
        elif reg_func_name == "Net":
            regr = GeneralLearner(
                lr=0.1,
                epochs=200,
                cost_func=nn.BCELoss(),
                in_shape=features_cal.shape[1],
                batch_size=128,
                model_type="deep_proba",
            )

        regr.fit(features_cal, yhat_cal_score)

        # compute error on holdout points
        features_orig = np.concatenate((a[:, np.newaxis], y), 1)
        output_orig = regr.predict(features_orig)

        if reg_func_name == "RF":
            est_orig_err = np.mean((yhat_score - output_orig) ** 2)
        elif reg_func_name == "Net":
            est_orig_err = (
                err_func(
                    torch.from_numpy(output_orig).float(), torch.from_numpy(yhat_score).float()
                )
                .detach()
                .cpu()
                .numpy()
            )

        # generate A and compare
        est_fake_err = np.zeros(num_p_val_rep)
        for inter_p_value in range(num_p_val_rep):
            random_array = np.random.uniform(low=0.0, high=1.0, size=a.shape)
            a_tilde = (random_array < p_success).astype(float)

            features_fake = np.concatenate((a_tilde[:, np.newaxis], y), 1)
            output_fake = regr.predict(features_fake)

            if reg_func_name == "RF":
                est_fake_err[inter_p_value] = np.mean((yhat_score - output_fake) ** 2)
            elif reg_func_name == "Net":
                est_fake_err[inter_p_value] = (
                    err_func(
                        torch.from_numpy(output_fake).float(), torch.from_numpy(yhat_score).float()
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

        p_val = 1.0 / (num_p_val_rep + 1) * (1 + sum(est_orig_err >= est_fake_err))

        test_i.append(p_val)

    print(
        "Fair dummies test (classification score), p-value:", np.mean(test_i)
    )  # should be uniform under ind.

    return test_i[0]
