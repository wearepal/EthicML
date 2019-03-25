"""
Implementation fo Zemel's Learned Fair Representations lifted from AIF360
https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/lfr.py
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import scipy.optimize as optim
from numba import jit

from ethicml.algorithms.algorithm_base import load_dataframe
from ethicml.algorithms.preprocess import PreAlgorithm
from ethicml.algorithms.utils import DataTuple


@jit
def distances(X, v, alpha, N, P, k):
    dists = np.zeros((N, P))
    for i in range(N):
        for p in range(P):
            for j in range(k):
                dists[i, j] += (X[i, p] - v[j, p]) * (X[i, p] - v[j, p]) * alpha[p]
    return dists


@jit
def M_nk(dists, N, k):
    M_nk = np.zeros((N, k))
    exp = np.zeros((N, k))
    denom = np.zeros(N)
    for i in range(N):
        for j in range(k):
            exp[i, j] = np.exp(-1 * dists[i, j])
            denom[i] += exp[i, j]
        for j in range(k):
            if denom[i]:
                M_nk[i, j] = exp[i, j] / denom[i]
            else:
                M_nk[i, j] = exp[i, j] / 1e-6
    return M_nk


@jit
def M_k(M_nk, N, k):
    M_k = np.zeros(k)
    for j in range(k):
        for i in range(N):
            M_k[j] += M_nk[i, j]
        M_k[j] /= N
    return M_k


@jit
def x_n_hat(X, M_nk, v, N, P, k):
    x_n_hat = np.zeros((N, P))
    L_x = 0.0
    for i in range(N):
        for p in range(P):
            for j in range(k):
                x_n_hat[i, p] += M_nk[i, j] * v[j, p]
            L_x += (X[i, p] - x_n_hat[i, p]) * (X[i, p] - x_n_hat[i, p])
    return x_n_hat, L_x


@jit
def yhat(M_nk, y, w, N, k):
    yhat = np.zeros(N)
    L_y = 0.0
    for i in range(N):
        for j in range(k):
            yhat[i] += M_nk[i, j] * w[j]
        yhat[i] = 1e-6 if yhat[i] <= 0 else yhat[i]
        yhat[i] = 0.999 if yhat[i] >= 1 else yhat[i]
        L_y += -1 * y[i] * np.log(yhat[i]) - (1.0 - y[i]) * np.log(1.0 - yhat[i])
    return yhat, L_y


@jit
def LFR_optim_obj(params, data_sensitive, data_nonsensitive, y_sensitive, y_nonsensitive,
                  k=10, A_x=0.01, A_y=0.1, A_z=0.5, results=0):
    LFR_optim_obj.iters += 1
    Ns, P = data_sensitive.shape
    Nns, _ = data_nonsensitive.shape

    alpha0 = params[:P]
    alpha1 = params[P : 2 * P]
    w = params[2 * P : (2 * P) + k]
    v = np.matrix(params[(2 * P) + k :]).reshape((k, P))

    dists_sensitive = distances(data_sensitive, v, alpha1, Ns, P, k)
    dists_nonsensitive = distances(data_nonsensitive, v, alpha0, Nns, P, k)

    M_nk_sensitive = M_nk(dists_sensitive, Ns, k)
    M_nk_nonsensitive = M_nk(dists_nonsensitive, Nns, k)

    M_k_sensitive = M_k(M_nk_sensitive, Ns, k)
    M_k_nonsensitive = M_k(M_nk_nonsensitive, Nns, k)

    L_z = 0.0
    for j in range(k):
        L_z += abs(M_k_sensitive[j] - M_k_nonsensitive[j])

    x_n_hat_sensitive, L_x1 = x_n_hat(data_sensitive, M_nk_sensitive, v, Ns, P, k)
    x_n_hat_nonsensitive, L_x2 = x_n_hat(
        data_nonsensitive, M_nk_nonsensitive, v, Nns, P, k
    )
    L_x = L_x1 + L_x2

    yhat_sensitive, L_y1 = yhat(M_nk_sensitive, y_sensitive, w, Ns, k)
    yhat_nonsensitive, L_y2 = yhat(M_nk_nonsensitive, y_nonsensitive, w, Nns, k)
    L_y = L_y1 + L_y2

    criterion = A_x * L_x + A_y * L_y + A_z * L_z

    if results:
        return yhat_sensitive, yhat_nonsensitive, M_nk_sensitive, M_nk_nonsensitive
    else:
        return criterion


LFR_optim_obj.iters = 0


class Zemel(PreAlgorithm):
    def __init__(self, threshold=0.5, clusters=2, Ax=0.01, Ay=0.1, Az=0.5,
                 max_iter=5000, maxfun=5000, epsilon=1e-5):
        self.k = clusters
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.threshold = threshold
        self.max_iter = max_iter
        self.maxfun = maxfun
        self.epsilon = epsilon

        # convert all parameter values to lists of strings
        flags: Dict[str, List[str]] = {
            'clusters': [str(clusters)],
            'Ax': [str(Ax)],
            'Ay': [str(Ay)],
            'Az': [str(Az)],
            'max_iter': [str(max_iter)],
            'maxfun': [str(maxfun)],
            'epsilon': [str(epsilon)],
            'threshold': [str(threshold)]
        }
        super().__init__(flags)

        np.random.seed(888)

    def _run(self, train: DataTuple, test: DataTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        features_dim = train.x.shape[1]

        sens_col = train.s.columns[0]
        training_sensitive = train.x[train.s[sens_col] == 0].to_numpy()
        training_nonsensitive = train.x[train.s[sens_col] == 1].to_numpy()
        ytrain_sensitive = train.y[train.s[sens_col] == 0].to_numpy()
        ytrain_nonsensitive = train.y[train.s[sens_col] == 1].to_numpy()

        model_inits = np.random.uniform(
            size=features_dim * 2 + self.k + features_dim * self.k
        )
        bnd: List[Tuple[Optional[int], Optional[int]]] = []
        for i, _ in enumerate(model_inits):
            if i < features_dim * 2 or i >= features_dim * 2 + self.k:
                bnd.append((None, None))
            else:
                bnd.append((0, 1))

        learned_model = optim.fmin_l_bfgs_b(
            LFR_optim_obj,
            x0=model_inits,
            epsilon=self.epsilon,
            args=(
                training_sensitive,
                training_nonsensitive,
                ytrain_sensitive[:, 0],
                ytrain_nonsensitive[:, 0],
                self.k,
                self.Ax,
                self.Ay,
                self.Az,
                0,
            ),
            bounds=bnd,
            approx_grad=True,
            maxfun=self.maxfun,
            maxiter=self.max_iter,
            disp=False,
        )[0]

        testing_sensitive = test.x[test.s[sens_col] == 0].to_numpy()
        testing_nonsensitive = test.x[test.s[sens_col] == 1].to_numpy()
        ytest_sensitive = test.y[test.s[sens_col] == 0].to_numpy()
        ytest_nonsensitive = test.y[test.s[sens_col] == 1].to_numpy()

        # Mutated, fairer dataset with new labels
        test_transformed = self.transform(testing_sensitive, testing_nonsensitive, ytest_sensitive, ytest_nonsensitive, learned_model, test)

        training_sensitive = train.x[train.s[sens_col] == 0].to_numpy()
        training_nonsensitive = train.x[train.s[sens_col] == 1].to_numpy()
        ytrain_sensitive = train.y[train.s[sens_col] == 0].to_numpy()
        ytrain_nonsensitive = train.y[train.s[sens_col] == 1].to_numpy()

        # extract training model parameters
        train_transformed = self.transform(training_sensitive, training_nonsensitive, ytrain_sensitive, ytrain_nonsensitive, learned_model, train)


        return train_transformed.x, test_transformed.x

    @property
    def name(self) -> str:
        return "Zemel"

    def transform(self, features_sens, features_nonsens, label_sens, label_nonsens, learned_model, dataset):
        Ns, P = features_sens.shape
        N, _ = features_nonsens.shape
        alphaoptim0 = learned_model[:P]
        alphaoptim1 = learned_model[P: 2 * P]
        woptim = learned_model[2 * P: (2 * P) + self.k]
        voptim = np.matrix(learned_model[(2 * P) + self.k:]).reshape((self.k, P))

        # compute distances on the test dataset using train model params
        dist_sensitive = distances(
            features_sens, voptim, alphaoptim1, Ns, P, self.k
        )
        dist_nonsensitive = distances(
            features_nonsens, voptim, alphaoptim0, N, P, self.k
        )

        # compute cluster probabilities for test instances
        M_nk_sensitive = M_nk(dist_sensitive, Ns, self.k)
        M_nk_nonsensitive = M_nk(dist_nonsensitive, N, self.k)

        # learned mappings for test instances
        res_sensitive = x_n_hat(
            features_sens, M_nk_sensitive, voptim, Ns, P, self.k
        )
        x_n_hat_sensitive = res_sensitive[0]
        res_nonsensitive = x_n_hat(
            features_nonsens, M_nk_nonsensitive, voptim, N, P, self.k
        )
        x_n_hat_nonsensitive = res_nonsensitive[0]

        # compute predictions for test instances
        res_sensitive = yhat(M_nk_sensitive, label_sens, woptim, Ns, self.k)
        y_hat_sensitive = res_sensitive[0]
        res_nonsensitive = yhat(
            M_nk_nonsensitive, label_nonsens, woptim, N, self.k
        )
        y_hat_nonsensitive = res_nonsensitive[0]

        sens_col = dataset.s.columns[0]

        sensitive_idx = dataset.x[dataset.s[sens_col] == 0].index
        nonsensitive_idx = dataset.x[dataset.s[sens_col] == 1].index

        transformed_features = np.zeros(shape=np.shape(dataset.x))
        transformed_labels = np.zeros(shape=np.shape(dataset.y))
        transformed_features[sensitive_idx] = x_n_hat_sensitive
        transformed_features[nonsensitive_idx] = x_n_hat_nonsensitive
        transformed_labels[sensitive_idx] = np.reshape(y_hat_sensitive, [-1, 1])
        transformed_labels[nonsensitive_idx] = np.reshape(y_hat_nonsensitive, [-1, 1])
        transformed_labels = (np.array(transformed_labels) > self.threshold).astype(
            np.float64
        )

        train_transformed = DataTuple(
            x=pd.DataFrame(transformed_features, columns=dataset.x.columns),
            s=pd.DataFrame(dataset.s, columns=dataset.s.columns),
            y=pd.DataFrame(transformed_labels, columns=dataset.y.columns),
        )

        return train_transformed


def load_data(flags):
    """Load data from the paths specified in the flags"""
    train = DataTuple(
        x=load_dataframe(Path(flags.train_x)),
        s=load_dataframe(Path(flags.train_s)),
        y=load_dataframe(Path(flags.train_y)),
    )
    test = DataTuple(
        x=load_dataframe(Path(flags.test_x)),
        s=load_dataframe(Path(flags.test_s)),
        y=load_dataframe(Path(flags.test_y)),
    )
    return train, test


def _parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser()

    # paths to the files with the data
    parser.add_argument("--train_x", required=True)
    parser.add_argument("--train_s", required=True)
    parser.add_argument("--train_y", required=True)
    parser.add_argument("--test_x", required=True)
    parser.add_argument("--test_s", required=True)
    parser.add_argument("--test_y", required=True)

    # paths to where the processed inputs should be stored
    parser.add_argument("--train_new", required=True)
    parser.add_argument("--test_new", required=True)

    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--Ax", type=float, required=True)
    parser.add_argument("--Ay", type=float, required=True)
    parser.add_argument("--Az", type=float, required=True)
    parser.add_argument("--max_iter", type=int, required=True)
    parser.add_argument("--maxfun", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    return parser.parse_args()


def main():
    """main method to run model"""
    flags = _parse_arguments()

    model = Zemel(clusters=flags.clusters,
                  Ax=flags.Ax,
                  Ay=flags.Ay,
                  Az=flags.Az,
                  max_iter=flags.max_iter,
                  maxfun=flags.maxfun,
                  epsilon=flags.epsilon,
                  threshold=flags.threshold)
    train, test = load_data(flags)
    model.save_transformations(model.run(train, test), flags)


if __name__ == "__main__":
    main()
