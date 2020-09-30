"""Make synthetic data for the 3rd way paper."""
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skewnorm
from sklearn import preprocessing

from ethicml import get_biased_and_debiased_subsets, save_label_plot
from ethicml.data.tabular_data.third_way import third_way


def make_x_bar(k: int, n: int) -> List[Tuple[float, float]]:
    """Initial potential."""
    # np.random.binomial(1, 1 / (j + 1), 1)[0]
    return [(0, 1 / (j + 1)) for j in range(k)]


def make_s(alpha: float, n=int):
    """Set S."""
    return np.random.binomial(1, alpha, n)


def make_dx(x_bar: np.ndarray, s: np.ndarray, gamma: float):
    """Skew the data replicating life experience."""
    for i, (mean, std) in enumerate(x_bar):

        def ske(s):
            return skewnorm(a=gamma * (2 * s - 1), loc=mean, scale=std).rvs(1)[0]

        # s[f"dx_{i}"] = 0
        s[f"dx_{i}"] = s["sens"].apply(ske)
        # s[s["sens"] == 1][f"dx_{i}"] = skewnorm(a=gamma, loc=mean, scale=std).rvs(
        #     s[s["sens"] == 1].shape[0]
        # )
        # skewnorm(mean, std, gamma * (2 * s)).rvs(1).round(2).clip(0, 0.95)

    def shortcut(s):
        return np.random.normal(loc=(2 * s - 1), scale=1.0)

    s[f"dx_{len(x_bar)}"] = s["sens"].apply(shortcut)
    return s


def make_x(dx_bar: pd.DataFrame, xi: float, n_bins: int):
    """Make observations of the data."""
    df = pd.DataFrame(
        {
            f"x_{i}": pd.cut(dx_bar[col], n_bins, labels=list(range(n_bins)))
            for i, col in enumerate(dx_bar.drop(["sens"], axis=1).sort_index(axis=1).columns)
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

    return pd.concat([pos_[dx_bar["sens"] == 1], neg_[dx_bar["sens"] == 0]], axis=0).sort_index()


def make_y(y_bar, s, beta):
    """Go from y_bar to Y."""
    threshold = y_bar.quantile([0.5]).values[0][0]
    print(threshold)

    def posify(x: pd.Series):
        new_x = x.copy()
        for i, val in x.iteritems():
            new_x.at[i] = (val > (threshold - np.random.binomial(1, beta, 1)[0])).astype(int)
        return new_x
        # return (new_x > threshold - (threshold * beta) * np.random.binomial(1, beta, 1)[0]).astype(
        #     int
        # )

    def negify(x: pd.Series):
        new_x = x.copy()
        for i, val in x.iteritems():
            new_x.at[i] = (val > (threshold + np.random.binomial(1, beta, 1)[0])).astype(int)
        return new_x
        # return (new_x > threshold + (threshold * beta) * np.random.binomial(1, beta, 1)[0]).astype(
        #     int
        # )

    pos_ = y_bar.apply(posify)
    neg_ = y_bar.apply(negify)
    return (
        pd.concat([pos_[s["sens"] == 1], neg_[s["sens"] == 0]], axis=0)
        .sort_index()
        .rename(columns={"y_bar": "y"})
    )


def main():
    """Make the 3rd way synthetic dataset."""
    np.random.seed(0)
    random.seed(0)
    samples = 10_000
    alpha = 0.65
    k = 5
    n_bins = 10
    beta = 0.25
    gamma = 0.75
    xi = 0.2

    x_bar = make_x_bar(k=k, n=samples)

    s = make_s(alpha=alpha, n=samples)
    s_df = pd.DataFrame(s, columns=["sens"])

    dx = make_dx(x_bar, s_df, gamma=gamma)

    x = make_x(dx, xi=xi, n_bins=n_bins)

    w = np.random.normal(0, 1, x.shape[1])

    x_ = x.values  # returns a numpy array
    min_max_scaler = preprocessing.StandardScaler()

    y_bar = pd.DataFrame(
        [1 / (1 + np.exp(np.dot(w, _x))) for _x in min_max_scaler.fit_transform(x_)],
        columns=["y_bar"],
    )

    y = make_y(y_bar, s_df, beta=beta)

    data = pd.concat([dx, x, y_bar, y], axis=1).sort_index()

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

    for i in range(4, 5):
        sns.distplot(data[data["sens"] == 1][f"dx_{i}"], color=colours[i])
        sns.distplot(data[data["sens"] == 0][f"dx_{i}"], color=colours[k + i])
    plt.show()

    for i in range(4, 5):
        sns.distplot(data[data["sens"] == 1][f"x_{i}"], color=colours[i])
        sns.distplot(data[data["sens"] == 0][f"x_{i}"], color=colours[k + i])
    plt.show()

    data = data.sample(frac=1.0).reset_index(drop=True)
    data.to_csv("./3rd.csv", index=False)


if __name__ == "__main__":
    main()

    data = third_way().load()
    save_label_plot(data, "./data_original.png")
    biased, unbiased = get_biased_and_debiased_subsets(data, mixing_factor=0.5, unbiased_pcnt=0.2)
    save_label_plot(biased, "./data_biased.png")
    save_label_plot(unbiased, "./data_unbiased.png")
