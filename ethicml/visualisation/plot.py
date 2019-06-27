"""
creates plots of a dataset
"""


import os
from pathlib import Path
from typing import Tuple, List

import imageio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from ethicml.utility.data_structures import DataTuple

MARKERS = ["s", "p", "P", "*", "+", "x", "o", "v"]


def save_2d_plot(data: DataTuple, filepath: str):
    """

    Args:
        data:
        filepath:
    """
    file_path = Path(filepath)
    columns = data.x.columns

    amalgamated = pd.concat([data.x, data.s, data.y], axis="columns")

    plot = sns.scatterplot(
        x=columns[0],
        y=columns[1],
        hue=data.y.columns[0],
        palette="Set2",
        data=amalgamated,
        style=data.s.columns[0],
        legend="full",
    )

    if not os.path.exists(file_path.parent):
        os.mkdir(file_path.parent)
    plot.figure.savefig(file_path)
    plt.clf()


def save_jointplot(data: DataTuple, filepath: str, dims: Tuple[int, int] = (0, 1)):
    """

    Args:
        data:
        filepath:
        dims:
    """
    file_path = Path(filepath)
    columns = data.x.columns

    amalgamated = pd.concat([data.x, data.y], axis="columns")

    plot = sns.jointplot(x=columns[dims[0]], y=columns[dims[1]], data=amalgamated, kind="kde")

    if not os.path.exists(file_path.parent):
        os.mkdir(file_path)
    plot.savefig(file_path)
    plt.clf()


def make_gif(files: List[str], name="movie"):
    """

    Args:
        files:
        name:
    """
    images = []
    for filename in files:
        images.append(imageio.imread(filename))
    imageio.mimsave(f"{name}.gif", images)


def save_label_plot(data: DataTuple, filename: str) -> None:
    """

    Args:
        data:
        filename:
    """
    file_path = Path(filename)

    # Only consider 1 sens attr for now
    s_col = data.s.columns[0]
    s_values = data.s[s_col].value_counts() / data.s[s_col].count()

    s_0_val = s_values[0]
    s_1_val = s_values[1]

    s_0_label = s_values.index[0]
    s_1_label = s_values.index[1]

    y_col = data.y.columns[0]
    y_s0 = (
        data.y[y_col][data.s[s_col] == 0].value_counts() / data.y[y_col][data.s[s_col] == 0].count()
    )
    y_s1 = (
        data.y[y_col][data.s[s_col] == 1].value_counts() / data.y[y_col][data.s[s_col] == 1].count()
    )

    y_0_label = y_s0.index[0]
    y_1_label = y_s0.index[1]

    mpl.style.use("seaborn-pastel")
    # plt.xkcd()

    # fig, ax = plt.subplots()

    _ = plt.bar(
        0,
        height=y_s0[y_0_label] * 100,
        width=s_0_val * 100,
        align="edge",
        edgecolor="black",
        color="C0",
    )
    _ = plt.bar(
        s_0_val * 100,
        height=y_s1[y_0_label] * 100,
        width=s_1_val * 100,
        align="edge",
        edgecolor="black",
        color="C1",
    )
    _ = plt.bar(
        0,
        height=y_s0[y_1_label] * 100,
        width=s_0_val * 100,
        bottom=y_s0[y_0_label] * 100,
        align="edge",
        edgecolor="black",
        color="C2",
    )
    _ = plt.bar(
        s_0_val * 100,
        height=y_s1[y_1_label] * 100,
        width=s_1_val * 100,
        bottom=y_s1[y_0_label] * 100,
        align="edge",
        edgecolor="black",
        color="C3",
    )

    plt.ylim([0, 100])
    plt.xlim([0, 100])
    plt.ylabel(f"Percent {y_col}=y")
    plt.xlabel(f"Percent {s_col}=s")
    plt.title("Dataset Composition by class and sensitive attribute")

    plt.legend(
        [
            f"y={y_0_label}, s={s_0_label}",
            f"y={y_0_label}, s={s_1_label}",
            f"y={y_1_label}, s={s_0_label}",
            f"y={y_1_label}, s={s_1_label}",
        ]
    )

    if not os.path.exists(file_path.parent):
        os.mkdir(file_path)
    plt.savefig(file_path)
    plt.clf()
