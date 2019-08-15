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

from ethicml.metrics import Metric
from ethicml.utility.data_structures import DataTuple
from ethicml.visualisation.common import errorbox, DataEntry, PlotDef

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


def plot_mean_std_box(results: pd.DataFrame, metric_1: Metric, metric_2: Metric) -> None:
    """

    Args:
        results:
        metric_1:
        metric_2:

    Returns:

    """
    directory = Path(".") / "plots"
    if not os.path.exists(directory):
        os.mkdir(directory)

    import itertools

    cols_met_1 = [col for col in results.columns if metric_1.name in col]
    cols_met_2 = [col for col in results.columns if metric_2.name in col]

    if len(cols_met_1) > 1:
        cols_met_1 = [col for col in cols_met_1 if ("-" in col) or ("/" in col)]
    if len(cols_met_2) > 1:
        cols_met_2 = [col for col in cols_met_2 if ("-" in col) or ("/" in col)]

    cols_met_1 += cols_met_2

    possible_pairs = list(itertools.combinations([col for col in cols_met_1], 2))

    for dataset in results.index.to_frame()['dataset'].unique():

        dataset_: str = str(dataset)

        for transform in results.index.to_frame()['transform'].unique():

            for pair in possible_pairs:

                entries: List[DataEntry] = []
                count = 0
                fig: plt.Figure
                plot: plt.Axes
                fig, plot = plt.subplots(dpi=300)
                plot_def = PlotDef(title=f"{dataset_} {transform}", entries=entries)

                mask_for_dataset = results.index.to_frame()['dataset'] == dataset_
                mask_for_transform = results.index.to_frame()['transform'] == transform
                if (
                    not pd.isnull(results.loc[mask_for_dataset & mask_for_transform][list(pair)])
                    .any()
                    .any()
                ):

                    for model in sorted(results.index.to_frame()['model'].unique()):

                        entries.append(
                            DataEntry(
                                f" {model} {transform}",
                                results.loc[
                                    mask_for_dataset
                                    & (results.index.to_frame()['model'] == model)
                                    & mask_for_transform
                                ],
                                count % 2 == 0,
                            )
                        )
                        count += 1

                    errorbox(
                        plot,
                        plot_def,
                        (pair[1], pair[1].replace("_", " ")),
                        (pair[0], pair[0].replace("_", " ")),
                        legend='outside',
                    )

                    metric_a = pair[0].replace("/", "_over_")
                    metric_b = pair[1].replace("/", "_over_")
                    fig.savefig(
                        directory / f"{dataset} {transform} {metric_a} {metric_b}.pdf",
                        bbox_inches='tight',
                    )
                    plt.cla()
                plt.close(fig)
