"""
creates plots of a dataset
"""
from pathlib import Path
from typing import Tuple, List, Union
import itertools

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

    file_path.parent.mkdir(exist_ok=True)
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

    file_path.parent.mkdir(exist_ok=True)
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

    file_path.parent.mkdir(exist_ok=True)
    plt.savefig(file_path)
    plt.clf()


def plot_single_mean_std_box(
    plot: plt.Axes, results: pd.DataFrame, x_axis: str, y_axis: str, dataset: str, transform: str
):
    """This is essentially a wrapper around the `errorbox` function

    This function can also be used to create figures with multiple plots on them, because it does
    not generate a Figure object itself.

    Args:
        plot: Plot object
        results: DataFrame with the data
        x_axis: name of metric for the x-axis
        y_axis: name of metric for the y-axis
        dataset: string that identifies the dataset
        transform: string that identifies the preprocessing method

    Returns
        the legend object if something was plotted
    """
    mask_for_dataset = results.index.to_frame()['dataset'] == dataset
    mask_for_transform = results.index.to_frame()['transform'] == transform
    matching_results = results.loc[mask_for_dataset & mask_for_transform]

    if not pd.isnull(matching_results[[x_axis, y_axis]]).any().any():
        entries: List[DataEntry] = []
        for count, model in enumerate(sorted(results.index.to_frame()['model'].unique())):
            mask_for_model = results.index.to_frame()['model'] == model
            data = results.loc[mask_for_dataset & mask_for_model & mask_for_transform]
            entries.append(DataEntry(f" {model} {transform}", data, count % 2 == 0))

        return errorbox(
            plot,
            plot_def=PlotDef(title=f"{dataset} {transform}", entries=entries),
            xaxis=(x_axis, x_axis.replace("_", " ")),
            yaxis=(y_axis, y_axis.replace("_", " ")),
            legend='outside',
        )
    return None


def plot_mean_std_box(
    results: pd.DataFrame,
    metric_y: Union[str, Metric],
    metric_x: Union[str, Metric],
    save: bool = True,
) -> List[Tuple[plt.Figure, plt.Axes]]:
    """Plot the given result with boxes that represent mean and standard deviation

    Args:
        results: a DataFrame that already contains the values of the metrics
        metric_y: a Metric object or a column name that defines which metric to plot on the y-axis
        metric_x: a Metric object or a column name that defines which metric to plot on the x-axis
        save: if True, save the plot as a PDF

    Returns:
        A list of all figures and plots
    """
    directory = Path(".") / "plots"
    directory.mkdir(exist_ok=True)

    # if the metric is given as a Metric object, take the name
    metric_x_name: str = metric_x.name if isinstance(metric_x, Metric) else metric_x
    metric_y_name: str = metric_y.name if isinstance(metric_y, Metric) else metric_y

    cols_x = [col for col in results.columns if metric_x_name in col]
    cols_y = [col for col in results.columns if metric_y_name in col]
    if not cols_y or not cols_x:
        raise ValueError(f"No matching columns found for either {metric_x_name} or {metric_y_name}")

    # if there are multiple matches, then the metric was `per_sensitive_attribute`. in this case,
    # we *only* want ratios and differences; not the plain result
    if len(cols_y) > 1:
        cols_y = [col for col in cols_y if ("-" in col) or ("/" in col)]
    if len(cols_x) > 1:
        cols_x = [col for col in cols_x if ("-" in col) or ("/" in col)]

    # generate the Cartesian product of `cols_x` and `cols_y`; i.e. all possible combinations
    possible_pairs = list(itertools.product(cols_x, cols_y))

    figure_list: List[Tuple[plt.Figure, plt.Axes]] = []
    for dataset in results.index.to_frame()['dataset'].unique():
        dataset_: str = str(dataset)
        for transform in results.index.to_frame()['transform'].unique():
            transform_: str = str(transform)
            for x_axis, y_axis in possible_pairs:
                fig: plt.Figure
                plot: plt.Axes
                fig, plot = plt.subplots(dpi=300)

                plot_single_mean_std_box(plot, results, x_axis, y_axis, dataset_, transform_)

                if save:
                    metric_a = x_axis.replace("/", "_over_")
                    metric_b = y_axis.replace("/", "_over_")
                    fig.savefig(
                        directory / f"{dataset} {transform} {metric_a} {metric_b}.pdf",
                        bbox_inches='tight',
                    )
                plt.close(fig)
                figure_list += [(fig, plot)]
    return figure_list
