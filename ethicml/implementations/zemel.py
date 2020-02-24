"""Implementation fo Zemel's Learned Fair Representations lifted from AIF360.

https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/lfr.py
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize as optim

from ethicml.utility import DataTuple, TestTuple

from .utils import PreAlgoArgs, load_data_from_flags, save_transformations

# Disable pylint's naming convention complaints - this code wasn't implemented by us
# pylint: disable=invalid-name


class ZemelArgs(PreAlgoArgs):
    """Arguments for the Zemel algorithm."""

    clusters: int
    Ax: float
    Ay: float
    Az: float
    max_iter: int
    maxfun: int
    epsilon: float
    threshold: float


def distances(x: np.ndarray, v: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compute distances.

    Compute the l2 distance between each feature, x_n, and each
    prototype vector, v_k, weighted by a feature-wise weight parameter, alpha
    Args:
        x: Input features
        v: Prototype vectors
        alpha: Individual weight parameters for each feature dimension

    Returns: array of shape [n, k]containing the weighted l2 distances between
    each feature, x_n, and each prototype vector, v_k
    """
    dists = np.sum(np.square(x[:, None] - v[None]) * alpha, axis=-1)
    return dists


def softmax(dists: np.ndarray) -> np.ndarray:
    """Compute the probability that x_n maps to a given prototype vector, v_k.

    Args:
        dists: l2 distances between each input feature and each prototype vector

    Returns:
        Membership probabilities of each input feature to each prototype
    """
    exp = np.exp(-dists)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom[denom == 0] = 1e-6
    matrix_nk = exp / denom

    return matrix_nk


def x_n_hat(x, matrix_nk, v) -> Tuple[np.ndarray, float]:
    """Reconstruct x_n from z_n.

    And compute the difference between the original input and its reconstruction using l2 loss.

    Args:
        x: Input features
        matrix_nk: Membership probabilities of each input feature to each prototype
        v: Prototype vectors

    Returns:
        Reconstruction of x and the l2 loss between the reconstruction and x
    """
    x_n_hat_obj = matrix_nk @ v
    l_x = (np.square(x - x_n_hat_obj)).sum()

    return x_n_hat_obj, l_x


def yhat(matrix_nk: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, float]:
    """Predict y and compute the cross-entropy loss.

     This is between the resulting predictions and the target labels

    Args:
        matrix_nk: Matrix encoding the membership probabilities of each input feature to
        each prototype
        y: target labels for each sample
        w: learned parameters governing the mapping from the prototypes to classification decisions

    Returns:
        Predictions and associated cross-entropy loss
    """
    y_hat = yhat_without_loss(matrix_nk, w)
    l_y = (-y * np.log(y_hat) - (1.0 - y) * np.log(1.0 - y_hat)).sum()

    return y_hat, l_y


def yhat_without_loss(matrix_nk: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict y without computing the classification loss.

    Args:
        matrix_nk: Matrix encoding the membership probabilities of each input feature to
        each prototype
        w: learned parameters governing the mapping from the prototypes to classification decisions

    Returns:
        Predictions and associated cross-entropy loss
    """
    y_hat = matrix_nk @ w
    y_hat[y_hat <= 0] = 1e-6
    y_hat[y_hat >= 1] = 0.999

    return y_hat


def lfr_optim_ob(
    params: np.ndarray,
    data_sensitive: np.ndarray,
    data_nonsensitive: np.ndarray,
    y_sensitive: np.ndarray,
    y_nonsensitive: np.ndarray,
    k: int = 10,
    a_x: float = 0.01,
    a_y: float = 0.1,
    a_z: float = 0.5,
    results: bool = False,
) -> Union[Tuple[np.ndarray, ...], float]:
    """Apply L-BFGS to minimize the LFR objective function.

    Args:
        params: Array of trainable parameters.
        data_sensitive: Data for the subset of individuals that are members
        of the protected set (i.e. S= 1)
        data_nonsensitive: Data for the subset of individuals that are not members
        of the protected set (i.e. S= 0)
        y_sensitive:
        y_nonsensitive:
        k: Number of prototype sets
        a_x: Pre-factor for the L_x loss term.
        a_y: Pre-factor for the L_y loss term.
        a_z: Pre-factor for the L_z loss term.
        results: Whether to return the predictions and prototype membership probabilities (True)
        or the weighted LFR loss (False)

    Returns:
        If 'results' is True, returns the predictions and prototype membership probabilities;
        otherwise returns the weighted LFR loss
    """
    lfr_optim_ob.iters += 1  # type: ignore[attr-defined]
    _, p = data_sensitive.shape

    alpha0 = params[:p]
    alpha1 = params[p : 2 * p]
    w = params[2 * p : (2 * p) + k]
    v = np.array(params[(2 * p) + k :]).reshape((k, p))

    dists_sensitive = distances(data_sensitive, v, alpha1)
    dists_nonsensitive = distances(data_nonsensitive, v, alpha0)

    m_nk_sensitive = softmax(dists_sensitive)
    m_nk_nonsensitive = softmax(dists_nonsensitive)

    m_k_sensitive = np.mean(m_nk_sensitive, axis=0)
    m_k_nonsensitive = np.mean(m_nk_nonsensitive, axis=0)

    # Loss term enforcing group fairness (minimizes MI between z and s)
    l_z = np.sum(np.abs(m_k_sensitive - m_k_nonsensitive))

    _, l_x1 = x_n_hat(data_sensitive, m_nk_sensitive, v)
    _, l_x2 = x_n_hat(data_nonsensitive, m_nk_nonsensitive, v)
    l_x = l_x1 + l_x2

    yhat_sensitive, l_y1 = yhat(m_nk_sensitive, y_sensitive, w)
    yhat_nonsensitive, l_y2 = yhat(m_nk_nonsensitive, y_nonsensitive, w)
    l_y = l_y1 + l_y2

    criterion = a_x * l_x + a_y * l_y + a_z * l_z

    return_tuple = (yhat_sensitive, yhat_nonsensitive, m_nk_sensitive, m_nk_nonsensitive)

    return return_tuple if results else criterion


lfr_optim_ob.iters = 0  # type: ignore[attr-defined]


def train_and_transform(
    train: DataTuple, test: TestTuple, flags: ZemelArgs
) -> (Tuple[DataTuple, TestTuple]):
    """Train the Zemel model and return the transformed features of the train and test sets."""
    np.random.seed(888)
    features_dim = train.x.shape[1]

    sens_col = train.s.columns[0]
    training_sensitive = train.x.loc[train.s[sens_col] == 0].to_numpy()
    training_nonsensitive = train.x.loc[train.s[sens_col] == 1].to_numpy()
    ytrain_sensitive = train.y.loc[train.s[sens_col] == 0].to_numpy()
    ytrain_nonsensitive = train.y.loc[train.s[sens_col] == 1].to_numpy()

    model_inits = np.random.uniform(
        size=int(features_dim * 2 + flags.clusters + features_dim * flags.clusters)
    )
    bnd: List[Tuple[Optional[int], Optional[int]]] = []
    for i, _ in enumerate(model_inits):
        if i < features_dim * 2 or i >= features_dim * 2 + flags.clusters:
            bnd.append((None, None))
        else:
            bnd.append((0, 1))

    learned_model = optim.fmin_l_bfgs_b(
        lfr_optim_ob,
        x0=model_inits,
        epsilon=flags.epsilon,
        args=(
            training_sensitive,
            training_nonsensitive,
            ytrain_sensitive[:, 0],
            ytrain_nonsensitive[:, 0],
            flags.clusters,
            flags.Ax,
            flags.Ay,
            flags.Az,
            0,
        ),
        bounds=bnd,
        approx_grad=True,
        maxfun=flags.maxfun,
        maxiter=flags.max_iter,
        disp=False,
    )[0]

    testing_sensitive = test.x.loc[test.s[sens_col] == 0].to_numpy()
    testing_nonsensitive = test.x.loc[test.s[sens_col] == 1].to_numpy()

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


def transform(features_sens, features_nonsens, learned_model, dataset, flags: ZemelArgs):
    """Transform a dataset.

    Args:
        features_sens: Sensitive features.
        features_nonsens: Nonsensitive features.
        label_sens: Sensitive labels.
        label_nonsens: Class labels.
        learned_model: Model optimized for the LFR objective.
        dataset: Dataset to be transformed.

    Returns:
        Dataframe of transformed features.
    """
    k = flags.clusters
    _, p = features_sens.shape
    alphaoptim0 = learned_model[:p]
    alphaoptim1 = learned_model[p : 2 * p]
    voptim = np.array(learned_model[(2 * p) + k :]).reshape((k, p))

    # compute distances on the test dataset using train model params
    dist_sensitive = distances(features_sens, voptim, alphaoptim1)
    dist_nonsensitive = distances(features_nonsens, voptim, alphaoptim0)

    # compute cluster probabilities for test instances
    m_nk_sensitive = softmax(dist_sensitive)
    m_nk_nonsensitive = softmax(dist_nonsensitive)

    # learned mappings for test instances
    res_sensitive = x_n_hat(features_sens, m_nk_sensitive, voptim)
    x_n_hat_sensitive = res_sensitive[0]
    res_nonsensitive = x_n_hat(features_nonsens, m_nk_nonsensitive, voptim)
    x_n_hat_nonsensitive = res_nonsensitive[0]

    sens_col = dataset.s.columns[0]

    sensitive_idx = dataset.x[dataset.s[sens_col] == 0].index
    nonsensitive_idx = dataset.x[dataset.s[sens_col] == 1].index

    transformed_features = np.zeros_like(dataset.x.to_numpy())
    transformed_features[sensitive_idx] = x_n_hat_sensitive
    transformed_features[nonsensitive_idx] = x_n_hat_nonsensitive

    return pd.DataFrame(transformed_features, columns=dataset.x.columns)


def main():
    """Main method to run model."""
    args = ZemelArgs()
    args.parse_args()

    train, test = load_data_from_flags(args)
    save_transformations(train_and_transform(train, test, args), args)


if __name__ == "__main__":
    main()
