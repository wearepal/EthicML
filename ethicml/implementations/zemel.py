"""Zemel algorithm."""
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.optimize as optim
from scipy.spatial.distance import cdist
from scipy.special import softmax

from ethicml.implementations.utils import PreAlgoArgs, load_data_from_flags, save_transformations
from ethicml.utility import DataTuple, TestTuple


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
    seed: int


def LFR_optim_objective(
    parameters: np.ndarray,
    x_unprivileged: np.ndarray,
    x_privileged: np.ndarray,
    y_unprivileged: np.ndarray,
    y_privileged: np.ndarray,
    clusters: int,
    A_x: float,
    A_y: float,
    A_z: float,
    print_interval,
    verbose,
) -> np.number:
    """LFR optim objective."""
    num_unprivileged, features_dim = x_unprivileged.shape
    num_privileged, _ = x_privileged.shape

    w = parameters[:clusters]
    prototypes = parameters[clusters:].reshape((clusters, features_dim))

    M_unprivileged, x_hat_unprivileged, y_hat_unprivileged = get_xhat_y_hat(
        prototypes, w, x_unprivileged
    )

    M_privileged, x_hat_privileged, y_hat_privileged = get_xhat_y_hat(prototypes, w, x_privileged)

    y_hat = np.concatenate([y_hat_unprivileged, y_hat_privileged], axis=0)
    y = np.concatenate([y_unprivileged.reshape((-1, 1)), y_privileged.reshape((-1, 1))], axis=0)

    L_x = np.mean((x_hat_unprivileged - x_unprivileged) ** 2) + np.mean(
        (x_hat_privileged - x_privileged) ** 2
    )
    L_z = np.mean(abs(np.mean(M_unprivileged, axis=0) - np.mean(M_privileged, axis=0)))
    L_y = -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))

    total_loss = A_x * L_x + A_y * L_y + A_z * L_z

    if verbose and LFR_optim_objective.steps % print_interval == 0:  # type: ignore[attr-defined]
        print(
            f"step: {LFR_optim_objective.steps}, "  # type: ignore[attr-defined]
            f"loss: {total_loss}, "
            f"L_x: {L_x},  "
            f"L_y: {L_y},  "
            f"L_z: {L_z}"
        )
    LFR_optim_objective.steps += 1  # type: ignore[attr-defined]

    return total_loss


def get_xhat_y_hat(
    prototypes: np.ndarray, w: np.ndarray, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get xhat y hat."""
    M = softmax(-cdist(x, prototypes), axis=1)
    x_hat = np.matmul(M, prototypes)
    y_hat = np.clip(
        np.matmul(M, w.reshape((-1, 1))), np.finfo(float).eps, 1.0 - np.finfo(float).eps
    )
    return M, x_hat, y_hat


def train_and_transform(
    train: DataTuple, test: TestTuple, flags: ZemelArgs
) -> (Tuple[DataTuple, TestTuple]):
    """Train the Zemel model and return the transformed features of the train and test sets."""
    np.random.seed(flags.seed)

    sens_col = train.s.columns[0]
    training_sensitive = train.x.loc[train.s[sens_col] == 0].to_numpy()
    training_nonsensitive = train.x.loc[train.s[sens_col] == 1].to_numpy()
    ytrain_sensitive = train.y.loc[train.s[sens_col] == 0].to_numpy()
    ytrain_nonsensitive = train.y.loc[train.s[sens_col] == 1].to_numpy()

    print_interval = 100
    verbose = False

    num_train_samples, features_dim = train.x.shape

    # Initialize the LFR optim objective parameters
    parameters_initialization = np.random.uniform(
        size=flags.clusters + features_dim * flags.clusters
    )
    bnd = [(0, 1)] * flags.clusters + [(None, None)] * features_dim * flags.clusters  # type: ignore[operator]
    LFR_optim_objective.steps = 0  # type: ignore[attr-defined]

    learned_model = optim.fmin_l_bfgs_b(
        LFR_optim_objective,
        x0=parameters_initialization,
        epsilon=1e-5,
        args=(
            training_nonsensitive,
            training_sensitive,
            ytrain_nonsensitive[:, 0],
            ytrain_sensitive[:, 0],
            flags.clusters,
            flags.Ax,
            flags.Ay,
            flags.Az,
            print_interval,
            verbose,
        ),
        bounds=bnd,
        approx_grad=True,
        maxfun=flags.maxfun,
        maxiter=flags.max_iter,
        disp=verbose,
    )[0]
    w = learned_model[: flags.clusters]
    prototypes = learned_model[flags.clusters :].reshape((flags.clusters, features_dim))

    testing_sensitive = test.x.loc[test.s[sens_col] == 0].to_numpy()
    testing_nonsensitive = test.x.loc[test.s[sens_col] == 1].to_numpy()

    train_transformed = trans(prototypes, w, training_nonsensitive, training_sensitive, train)
    test_transformed = trans(prototypes, w, testing_nonsensitive, testing_sensitive, test)

    return (
        DataTuple(x=train_transformed, s=train.s, y=train.y, name=train.name),
        TestTuple(x=test_transformed, s=test.s, name=test.name),
    )


def trans(prototypes, w, nonsens, sens, dataset):
    """Trans."""
    _, features_hat_nonsensitive, labels_hat_nonsensitive = get_xhat_y_hat(prototypes, w, nonsens)

    _, features_hat_sensitive, labels_hat_sensitive = get_xhat_y_hat(prototypes, w, sens)

    sens_col = dataset.s.columns[0]

    sensitive_idx = dataset.x[dataset.s[sens_col] == 0].index
    nonsensitive_idx = dataset.x[dataset.s[sens_col] == 1].index

    transformed_features = np.zeros_like(dataset.x.to_numpy())
    transformed_features[sensitive_idx] = features_hat_sensitive
    transformed_features[nonsensitive_idx] = features_hat_nonsensitive

    return pd.DataFrame(transformed_features, columns=dataset.x.columns)


def main() -> None:
    """Main method to run model."""
    args = ZemelArgs()
    args.parse_args()

    train, test = load_data_from_flags(args)
    save_transformations(train_and_transform(train, test, args), args)


if __name__ == "__main__":
    """Learning fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [2]_.
    References:
        .. [2] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,  "Learning
           Fair Representations." International Conference on Machine Learning,
           2013.
    Based on code from https://github.com/zjelveh/learning-fair-representations
    Which in turn, we've got from AIF360
    """
    main()
