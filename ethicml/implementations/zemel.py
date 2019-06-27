"""
Implementation fo Zemel's Learned Fair Representations lifted from AIF360
https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/lfr.py
"""
from typing import List, Tuple, Dict, Optional, Union

import pandas as pd
import numpy as np
import scipy.optimize as optim
from numba import jit

from ethicml.utility.data_structures import DataTuple, TestTuple
from ethicml.implementations.utils import (
    load_data_from_flags,
    save_transformations,
    pre_algo_argparser,
)

# Disable pylint's naming convention complaints - this code wasn't implemented by us
# pylint: disable=invalid-name


@jit
def distances(X, v, alpha, N, P, k):
    """
    Calculates distances?
    Args:
        X:
        v:
        alpha:
        N:
        P:
        k:

    Returns:

    """
    dists = np.zeros((N, P))
    for i in range(N):
        for p in range(P):
            for j in range(k):
                dists[i, j] += (X[i, p] - v[j, p]) * (X[i, p] - v[j, p]) * alpha[p]
    return dists


@jit
def M_nk(dists, N, k):
    """
    Not a clue
    Args:
        dists:
        N:
        k:

    Returns:

    """
    matrix_nk = np.zeros((N, k))
    exp = np.zeros((N, k))
    denom = np.zeros(N)
    for i in range(N):
        for j in range(k):
            exp[i, j] = np.exp(-1 * dists[i, j])
            denom[i] += exp[i, j]
        for j in range(k):
            if denom[i]:
                matrix_nk[i, j] = exp[i, j] / denom[i]
            else:
                matrix_nk[i, j] = exp[i, j] / 1e-6
    return matrix_nk


@jit
def M_k(matrix_nk, N, k):
    """
    ???
    Args:
        matrix_nk:
        N:
        k:

    Returns:

    """
    matrix_k = np.zeros(k)
    for j in range(k):
        for i in range(N):
            matrix_k[j] += matrix_nk[i, j]
        matrix_k[j] /= N
    return matrix_k


@jit
def x_n_hat(X, matrix_nk, v, N, P, k):
    """???"""
    x_n_hat_obj = np.zeros((N, P))
    L_x = 0.0
    for i in range(N):
        for p in range(P):
            for j in range(k):
                x_n_hat_obj[i, p] += matrix_nk[i, j] * v[j, p]
            L_x += (X[i, p] - x_n_hat_obj[i, p]) * (X[i, p] - x_n_hat_obj[i, p])
    return x_n_hat_obj, L_x


@jit
def yhat(matrix_nk, y, w, N, k):
    """no idea"""
    y_hat = np.zeros(N)
    L_y = 0.0
    for i in range(N):
        for j in range(k):
            y_hat[i] += matrix_nk[i, j] * w[j]
        y_hat[i] = 1e-6 if y_hat[i] <= 0 else y_hat[i]
        y_hat[i] = 0.999 if y_hat[i] >= 1 else y_hat[i]
        L_y += -1 * y[i] * np.log(y_hat[i]) - (1.0 - y[i]) * np.log(1.0 - y_hat[i])
    return y_hat, L_y


@jit
def yhat_without_loss(matrix_nk, w, N, k):
    """no idea without loss"""
    y_hat = np.zeros(N)
    for i in range(N):
        for j in range(k):
            y_hat[i] += matrix_nk[i, j] * w[j]
        y_hat[i] = 1e-6 if y_hat[i] <= 0 else y_hat[i]
        y_hat[i] = 0.999 if y_hat[i] >= 1 else y_hat[i]
    return y_hat


@jit
def LFR_optim_obj(
    params,
    data_sensitive,
    data_nonsensitive,
    y_sensitive,
    y_nonsensitive,
    k=10,
    A_x=0.01,
    A_y=0.1,
    A_z=0.5,
    results=0,
):
    """
    The funtion to be optimized
    Args:
        params:
        data_sensitive:
        data_nonsensitive:
        y_sensitive:
        y_nonsensitive:
        k:
        A_x:
        A_y:
        A_z:
        results:

    Returns:

    """
    LFR_optim_obj.iters += 1
    Ns, P = data_sensitive.shape
    Nns, _ = data_nonsensitive.shape

    alpha0 = params[:P]
    alpha1 = params[P : 2 * P]
    w = params[2 * P : (2 * P) + k]
    v = np.array(params[(2 * P) + k :]).reshape((k, P))

    dists_sensitive = distances(data_sensitive, v, alpha1, Ns, P, k)
    dists_nonsensitive = distances(data_nonsensitive, v, alpha0, Nns, P, k)

    M_nk_sensitive = M_nk(dists_sensitive, Ns, k)
    M_nk_nonsensitive = M_nk(dists_nonsensitive, Nns, k)

    M_k_sensitive = M_k(M_nk_sensitive, Ns, k)
    M_k_nonsensitive = M_k(M_nk_nonsensitive, Nns, k)

    L_z = 0.0
    for j in range(k):
        L_z += abs(M_k_sensitive[j] - M_k_nonsensitive[j])

    _, L_x1 = x_n_hat(data_sensitive, M_nk_sensitive, v, Ns, P, k)
    _, L_x2 = x_n_hat(data_nonsensitive, M_nk_nonsensitive, v, Nns, P, k)
    L_x = L_x1 + L_x2

    yhat_sensitive, L_y1 = yhat(M_nk_sensitive, y_sensitive, w, Ns, k)
    yhat_nonsensitive, L_y2 = yhat(M_nk_nonsensitive, y_nonsensitive, w, Nns, k)
    L_y = L_y1 + L_y2

    criterion = A_x * L_x + A_y * L_y + A_z * L_z

    return_tuple = (yhat_sensitive, yhat_nonsensitive, M_nk_sensitive, M_nk_nonsensitive)
    return return_tuple if results else criterion


LFR_optim_obj.iters = 0


def train_and_transform(
    train: DataTuple, test: TestTuple, flags: Dict[str, Union[int, float]]
) -> (Tuple[DataTuple, TestTuple]):
    """Train the Zemel model and return the transformed features of the train and test sets"""
    np.random.seed(888)
    features_dim = train.x.shape[1]

    sens_col = train.s.columns[0]
    training_sensitive = train.x.loc[train.s[sens_col] == 0].to_numpy()
    training_nonsensitive = train.x.loc[train.s[sens_col] == 1].to_numpy()
    ytrain_sensitive = train.y.loc[train.s[sens_col] == 0].to_numpy()
    ytrain_nonsensitive = train.y.loc[train.s[sens_col] == 1].to_numpy()

    model_inits = np.random.uniform(
        size=int(features_dim * 2 + flags['clusters'] + features_dim * flags['clusters'])
    )
    bnd: List[Tuple[Optional[int], Optional[int]]] = []
    for i, _ in enumerate(model_inits):
        if i < features_dim * 2 or i >= features_dim * 2 + flags['clusters']:
            bnd.append((None, None))
        else:
            bnd.append((0, 1))

    learned_model = optim.fmin_l_bfgs_b(
        LFR_optim_obj,
        x0=model_inits,
        epsilon=flags['epsilon'],
        args=(
            training_sensitive,
            training_nonsensitive,
            ytrain_sensitive[:, 0],
            ytrain_nonsensitive[:, 0],
            flags['clusters'],
            flags['Ax'],
            flags['Ay'],
            flags['Az'],
            0,
        ),
        bounds=bnd,
        approx_grad=True,
        maxfun=flags['maxfun'],
        maxiter=flags['max_iter'],
        disp=False,
    )[0]

    testing_sensitive = test.x.loc[test.s[sens_col] == 0].to_numpy()
    testing_nonsensitive = test.x.loc[test.s[sens_col] == 1].to_numpy()
    # ytest_sensitive = test.y.loc[test.s[sens_col] == 0].to_numpy()
    # ytest_nonsensitive = test.y.loc[test.s[sens_col] == 1].to_numpy()

    # Mutated, fairer dataset with new labels
    test_transformed = transform(
        testing_sensitive, testing_nonsensitive, learned_model, test, flags
    )

    training_sensitive = train.x.loc[train.s[sens_col] == 0].to_numpy()
    training_nonsensitive = train.x.loc[train.s[sens_col] == 1].to_numpy()
    ytrain_sensitive = train.y.loc[train.s[sens_col] == 0].to_numpy()
    ytrain_nonsensitive = train.y.loc[train.s[sens_col] == 1].to_numpy()

    # extract training model parameters
    train_transformed = transform(
        training_sensitive, training_nonsensitive, learned_model, train, flags
    )

    return (
        DataTuple(x=train_transformed, s=train.s, y=train.y, name=train.name),
        TestTuple(x=test_transformed, s=test.s, name=test.name),
    )


def transform(features_sens, features_nonsens, learned_model, dataset, flags):
    """
    transform a dataset based on the
    Args:
        features_sens:
        features_nonsens:
        label_sens:
        label_nonsens:
        learned_model:
        dataset:

    Returns:

    """
    k = flags['clusters']
    Ns, P = features_sens.shape
    N, _ = features_nonsens.shape
    alphaoptim0 = learned_model[:P]
    alphaoptim1 = learned_model[P : 2 * P]
    # woptim = learned_model[2 * P : (2 * P) + k]
    voptim = np.array(learned_model[(2 * P) + k :]).reshape((k, P))

    # compute distances on the test dataset using train model params
    dist_sensitive = distances(features_sens, voptim, alphaoptim1, Ns, P, k)
    dist_nonsensitive = distances(features_nonsens, voptim, alphaoptim0, N, P, k)

    # compute cluster probabilities for test instances
    M_nk_sensitive = M_nk(dist_sensitive, Ns, k)
    M_nk_nonsensitive = M_nk(dist_nonsensitive, N, k)

    # learned mappings for test instances
    res_sensitive = x_n_hat(features_sens, M_nk_sensitive, voptim, Ns, P, k)
    x_n_hat_sensitive = res_sensitive[0]
    res_nonsensitive = x_n_hat(features_nonsens, M_nk_nonsensitive, voptim, N, P, k)
    x_n_hat_nonsensitive = res_nonsensitive[0]

    # compute predictions for test instances
    # y_hat_sensitive = yhat_without_loss(M_nk_sensitive, woptim, Ns, k)
    # y_hat_nonsensitive = yhat_without_loss(M_nk_nonsensitive, woptim, N, k)

    sens_col = dataset.s.columns[0]

    sensitive_idx = dataset.x[dataset.s[sens_col] == 0].index
    nonsensitive_idx = dataset.x[dataset.s[sens_col] == 1].index

    transformed_features = np.zeros_like(dataset.x.values)
    # transformed_labels = np.zeros_like(dataset.y.values)
    transformed_features[sensitive_idx] = x_n_hat_sensitive
    transformed_features[nonsensitive_idx] = x_n_hat_nonsensitive
    # transformed_labels[sensitive_idx] = np.reshape(y_hat_sensitive, [-1, 1])
    # transformed_labels[nonsensitive_idx] = np.reshape(y_hat_nonsensitive, [-1, 1])
    # transformed_labels = (np.array(transformed_labels) > flags['threshold']).astype(np.float64)

    # return DataTuple(
    #     x=pd.DataFrame(transformed_features, columns=dataset.x.columns),
    #     s=pd.DataFrame(dataset.s, columns=dataset.s.columns),
    #     y=pd.DataFrame(transformed_labels, columns=dataset.y.columns),
    # )

    return pd.DataFrame(transformed_features, columns=dataset.x.columns)


def main():
    """main method to run model"""
    parser = pre_algo_argparser()
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--Ax", type=float, required=True)
    parser.add_argument("--Ay", type=float, required=True)
    parser.add_argument("--Az", type=float, required=True)
    parser.add_argument("--max_iter", type=int, required=True)
    parser.add_argument("--maxfun", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    flags = vars(parser.parse_args())

    train, test = load_data_from_flags(flags)
    save_transformations(train_and_transform(train, test, flags), args)


if __name__ == "__main__":
    main()
