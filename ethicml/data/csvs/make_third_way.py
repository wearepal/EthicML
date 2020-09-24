"""Make synthetic data for the 3rd way paper."""
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def make_x_bar(k: int, n: int) -> List[np.ndarray]:
    """Initial potential."""
    return np.array(
        [np.random.normal(np.random.binomial(1, 1 / (j + 1), 1), 1 / (j + 1), n) for j in range(k)]
    ).transpose()


def make_s(alpha: float, n=int):
    """Set S."""
    return np.random.binomial(1, alpha, n)


def make_dx(x_bar: np.ndarray, s: np.ndarray, gamma: float):
    """Skew the data replicating life experience."""
    return x_bar + gamma * ((s * 2) - 1)[:, None]


def make_x(dx_bar: np.ndarray, s: np.ndarray, xi: float, n_bins: int):
    """Make observations of the data."""
    df = pd.DataFrame(
        {
            f"x_{i}": pd.cut(dx_bar[col], n_bins, labels=list(range(n_bins)))
            for i, col in enumerate(dx_bar.columns)
        }
    )

    def posify(x: pd.Series):
        max = x.max()
        new_x = x.copy()
        for i, val in x.iteritems():
            if val < max:
                new_x.at[i] = int(val + np.random.binomial(1, xi, 1)[0])
        return new_x

    def negify(x: pd.Series):
        min = x.min()
        new_x = x.copy()
        for i, val in x.iteritems():
            if val > min:
                new_x.at[i] = int(val - np.random.binomial(1, xi, 1)[0])
        return new_x

    pos_ = df.apply(posify)
    neg_ = df.apply(negify)

    return pd.concat([pos_[s["sens"] == 1], neg_[s["sens"] == 0]], axis=0).sort_index()


def main():
    """Make the 3rd way synthetic dataset."""
    np.random.seed(0)
    random.seed(0)
    samples = 1_000
    k = 5
    n_bins = 15
    gamma = 1.0
    xi = 0.3

    x_bar = make_x_bar(k=k, n=samples)
    x_bar_df = pd.DataFrame(x_bar, columns=[f"x_bar_{i}" for i in range(k)])

    s = make_s(alpha=0.5, n=samples)
    s_df = pd.DataFrame(s, columns=["sens"])

    dx = make_dx(x_bar, s, gamma=gamma)
    dx_df = pd.DataFrame(dx, columns=[f"dx_bar_{i}" for i in range(k)])

    x = make_x(dx_df, s_df, xi=xi, n_bins=n_bins)

    data = pd.concat([x_bar_df, s_df, dx_df, x], axis=1)

    colours = [
        "black",
        "grey",
        "silver",
        "rosybrown",
        "firebrick",
        "lightpink",
        "lavenderblush",
        "orchid",
        "darkmagenta",
        "thistle",
    ]
    for i in range(k):
        sns.distplot(data[data["sens"] == 1][f"x_bar_{i}"], color=colours[i])
        sns.distplot(data[data["sens"] == 0][f"x_bar_{i}"], color=colours[k + i])
    plt.show()

    for i in range(k):
        sns.distplot(data[data["sens"] == 1][f"dx_bar_{i}"], color=colours[i])
        sns.distplot(data[data["sens"] == 0][f"dx_bar_{i}"], color=colours[k + i])
    plt.show()

    for i in range(k):
        sns.distplot(data[data["sens"] == 1][f"x_{i}"], color=colours[i])
        sns.distplot(data[data["sens"] == 0][f"x_{i}"], color=colours[k + i])
    plt.show()


if __name__ == "__main__":
    main()
