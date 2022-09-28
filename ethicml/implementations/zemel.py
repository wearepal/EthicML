"""Zemel algorithm."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, NamedTuple

from joblib import dump, load
import numpy as np
import pandas as pd
import scipy.optimize as optim
from scipy.spatial.distance import cdist
from scipy.special import softmax  # type: ignore[attr-defined]

from ethicml.implementations.utils import load_data_from_flags, save_transformations
from ethicml.utility import DataTuple, SubgroupTuple, TestTuple

if TYPE_CHECKING:
    from ethicml.models.preprocess.pre_subprocess import PreAlgoArgs
    from ethicml.models.preprocess.zemel import ZemelArgs


class Model(NamedTuple):
    """Model."""

    prototypes: np.ndarray
    w: np.ndarray


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
    print_interval: int,
    verbose: bool,
) -> np.number:
    """LFR optim objective."""
    _, features_dim = x_unprivileged.shape

    w = parameters[:clusters]
    prototypes = parameters[clusters:].reshape((clusters, features_dim))

    m_unprivileged, x_hat_unprivileged, y_hat_unprivileged = get_xhat_y_hat(
        prototypes, w, x_unprivileged
    )

    m_privileged, x_hat_privileged, y_hat_privileged = get_xhat_y_hat(prototypes, w, x_privileged)

    y_hat = np.concatenate([y_hat_unprivileged, y_hat_privileged], axis=0)
    y = np.concatenate([y_unprivileged.reshape((-1, 1)), y_privileged.reshape((-1, 1))], axis=0)

    l_x = np.mean((x_hat_unprivileged - x_unprivileged) ** 2) + np.mean(
        (x_hat_privileged - x_privileged) ** 2
    )
    l_z = np.mean(abs(np.mean(m_unprivileged, axis=0) - np.mean(m_privileged, axis=0)))
    l_y = -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))

    total_loss = A_x * l_x + A_y * l_y + A_z * l_z

    if verbose and LFR_optim_objective.steps % print_interval == 0:  # type: ignore[attr-defined]
        print(
            f"step: {LFR_optim_objective.steps}, "  # type: ignore[attr-defined]
            f"loss: {total_loss}, "
            f"L_x: {l_x},  "
            f"L_y: {l_y},  "
            f"L_z: {l_z}"
        )
    LFR_optim_objective.steps += 1  # type: ignore[attr-defined]

    return total_loss


def get_xhat_y_hat(
    prototypes: np.ndarray, w: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get xhat y hat."""
    M = softmax(-cdist(x, prototypes), axis=1)
    x_hat = np.matmul(M, prototypes)
    y_hat = np.clip(
        np.matmul(M, w.reshape((-1, 1))), np.finfo(float).eps, 1.0 - np.finfo(float).eps
    )
    return M, x_hat, y_hat


def train_and_transform(
    train: DataTuple, test: SubgroupTuple, flags: ZemelArgs, seed: int
) -> (tuple[DataTuple, SubgroupTuple]):
    """Train and transform."""
    prototypes, w = fit(train, flags, seed=seed)

    training_sensitive = train.x.loc[train.s == 0].to_numpy()
    training_nonsensitive = train.x.loc[train.s == 1].to_numpy()

    testing_sensitive = test.x.loc[test.s == 0].to_numpy()
    testing_nonsensitive = test.x.loc[test.s == 1].to_numpy()

    train_transformed = trans(prototypes, w, training_nonsensitive, training_sensitive, train)
    test_transformed = trans(prototypes, w, testing_nonsensitive, testing_sensitive, test)

    return train.replace(x=train_transformed), test.replace(x=test_transformed)


def transform(data: SubgroupTuple, prototypes: np.ndarray, w: np.ndarray) -> SubgroupTuple:
    """Transform."""
    data_sens = data.x.loc[data.s == 0].to_numpy()
    data_nons = data.x.loc[data.s == 1].to_numpy()
    transformed = trans(prototypes, w, data_nons, data_sens, data)
    return data.replace(x=transformed)


def fit(train: DataTuple, flags: ZemelArgs, seed: int) -> Model:
    """Train the Zemel model and return the transformed features of the train and test sets."""
    np.random.seed(seed)

    training_sensitive = train.x.loc[train.s == 0].to_numpy()
    training_nonsensitive = train.x.loc[train.s == 1].to_numpy()
    ytrain_sensitive = train.y.loc[train.s == 0].to_numpy()
    ytrain_nonsensitive = train.y.loc[train.s == 1].to_numpy()

    print_interval = 100
    verbose = False

    _, features_dim = train.x.shape

    # Initialize the LFR optim objective parameters
    parameters_initialization = np.random.uniform(
        size=flags["clusters"] + features_dim * flags["clusters"]
    )
    bnd = [(0, 1)] * flags["clusters"] + [(None, None)] * features_dim * flags["clusters"]
    LFR_optim_objective.steps = 0  # type: ignore[attr-defined]

    learned_model = optim.fmin_l_bfgs_b(  # type: ignore[attr-defined]
        LFR_optim_objective,
        x0=parameters_initialization,
        epsilon=1e-5,
        args=(
            training_nonsensitive,
            training_sensitive,
            ytrain_nonsensitive,
            ytrain_sensitive,
            flags["clusters"],
            flags["Ax"],
            flags["Ay"],
            flags["Az"],
            print_interval,
            verbose,
        ),
        bounds=bnd,
        approx_grad=True,
        maxfun=flags["maxfun"],
        maxiter=flags["max_iter"],
        disp=verbose,
    )[0]
    w = learned_model[: flags["clusters"]]
    prototypes = learned_model[flags["clusters"] :].reshape((flags["clusters"], features_dim))

    return Model(prototypes=prototypes, w=w)


def trans(
    prototypes: np.ndarray, w: np.ndarray, nonsens: np.ndarray, sens: np.ndarray, dataset: TestTuple
) -> pd.DataFrame:
    """Trans."""
    _, features_hat_nonsensitive, _ = get_xhat_y_hat(prototypes, w, nonsens)

    _, features_hat_sensitive, _ = get_xhat_y_hat(prototypes, w, sens)

    sensitive_idx = dataset.x[dataset.s == 0].index
    nonsensitive_idx = dataset.x[dataset.s == 1].index

    transformed_features = np.zeros_like(dataset.x.to_numpy())
    transformed_features[sensitive_idx] = features_hat_sensitive
    transformed_features[nonsensitive_idx] = features_hat_nonsensitive

    return pd.DataFrame(transformed_features, columns=dataset.x.columns)


def main() -> None:
    """LFR Model.

    Learning fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [2]_.

    References:
        .. [2] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,  "Learning
           Fair Representations." International Conference on Machine Learning,
           2013.

    Based on code from https://github.com/zjelveh/learning-fair-representations
    Which in turn, we've got from AIF360
    """
    pre_algo_args: PreAlgoArgs = json.loads(sys.argv[1])
    flags: ZemelArgs = json.loads(sys.argv[2])
    if pre_algo_args["mode"] == "run":
        train, test = load_data_from_flags(pre_algo_args)
        save_transformations(
            train_and_transform(train, test, flags, seed=pre_algo_args["seed"]), pre_algo_args
        )
    elif pre_algo_args["mode"] == "fit":
        train = DataTuple.from_file(Path(pre_algo_args["train"]))
        model = fit(train, flags, seed=pre_algo_args["seed"])
        training_sensitive = train.x.loc[train.s == 0].to_numpy()
        training_nonsensitive = train.x.loc[train.s == 1].to_numpy()
        train_transformed = trans(
            model.prototypes, model.w, training_nonsensitive, training_sensitive, train
        )
        data = train.replace(x=train_transformed)
        data.save_to_file(Path(pre_algo_args["new_train"]))
        dump(model, Path(pre_algo_args["model"]))
    elif pre_algo_args["mode"] == "transform":
        model = load(Path(pre_algo_args["model"]))
        test = SubgroupTuple.from_file(Path(pre_algo_args["test"]))
        transformed_test = transform(test, model.prototypes, model.w)
        transformed_test.save_to_file(Path(pre_algo_args["new_test"]))


if __name__ == "__main__":
    main()
