"""Fair Dummies utility functions."""
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KernelDensity

from ethicml.implementations.hgr_modules.utility_functions import compute_acc
from ethicml.implementations.pytorch_common import GeneralLearner


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


def calc_accuracy(
    outputs: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:  # Care outputs are going to be in dimension 2
    """Accuracy."""
    max_vals, max_indices = torch.max(outputs, 1)
    return (max_indices == y).sum().detach().cpu().numpy() / max_indices.size()[0]


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

    rng = np.random.default_rng(0)

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
            random_array = rng.uniform(low=0.0, high=1.0, size=a.shape)
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
    rng = np.random.default_rng(0)

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
            random_array = rng.uniform(low=0.0, high=1.0, size=a.shape)
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
