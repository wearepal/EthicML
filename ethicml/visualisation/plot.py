"""
creates plots of a dataset
"""
from pathlib import Path
from typing import Tuple, List, Union
import itertools

from typing_extensions import Literal
import imageio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from ethicml.metrics import Metric
from ethicml.utility.data_structures import DataTuple
from ethicml.visualisation.common import errorbox, DataEntry, PlotDef, LegendType

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

    fig, plot = plt.subplots()

    quadrant1 = plot.bar(
        0,
        height=y_s0[y_0_label] * 100,
        width=s_0_val * 100,
        align="edge",
        edgecolor="black",
        color="C0",
    )
    quadrant2 = plot.bar(
        s_0_val * 100,
        height=y_s1[y_0_label] * 100,
        width=s_1_val * 100,
        align="edge",
        edgecolor="black",
        color="C1",
    )
    quadrant3 = plot.bar(
        0,
        height=y_s0[y_1_label] * 100,
        width=s_0_val * 100,
        bottom=y_s0[y_0_label] * 100,
        align="edge",
        edgecolor="black",
        color="C2",
    )
    quadrant4 = plot.bar(
        s_0_val * 100,
        height=y_s1[y_1_label] * 100,
        width=s_1_val * 100,
        bottom=y_s1[y_0_label] * 100,
        align="edge",
        edgecolor="black",
        color="C3",
    )

    plot.set_ylim(0, 100)
    plot.set_xlim(0, 100)
    plot.set_ylabel(f"Percent {y_col}=y")
    plot.set_xlabel(f"Percent {s_col}=s")
    plot.set_title("Dataset Composition by class and sensitive attribute")

    plot.legend(
        [quadrant1, quadrant2, quadrant3, quadrant4],
        [
            f"y={y_0_label}, s={s_0_label}",
            f"y={y_0_label}, s={s_1_label}",
            f"y={y_1_label}, s={s_0_label}",
            f"y={y_1_label}, s={s_1_label}",
        ],
    )

    file_path.parent.mkdir(exist_ok=True)
    fig.savefig(file_path)


def single_plot_mean_std_box(
    plot: plt.Axes,
    results: pd.DataFrame,
    xaxis: Tuple[str, str],
    yaxis: Tuple[str, str],
    dataset: str,
    transform: str,
    legend: Union[None, LegendType, Tuple[LegendType, float]] = "outside",
    markersize: int = 6,
    use_cross: bool = False,
) -> Union[None, Literal[False], mpl.legend.Legend]:
    """This is essentially a wrapper around the `errorbox` function

    This function can also be used to create figures with multiple plots on them, because it does
    not generate a Figure object itself.

    Args:
        plot: Plot object
        results: DataFrame with the data
        x_axis: name of column that's plotted on the x-axis
        y_axis: name of column that's plotted on the y-axis
        dataset: string that identifies the dataset
        transform: string that identifies the preprocessing method

    Returns
        the legend object if something was plotted; False otherwise
    """
    mask_for_dataset = results.index.get_level_values('dataset') == dataset
    mask_for_transform = results.index.get_level_values('transform') == transform
    matching_results = results.loc[mask_for_dataset & mask_for_transform]

    if pd.isnull(matching_results[[xaxis[0], yaxis[0]]]).any().any():
        return False  # nothing to plot

    entries: List[DataEntry] = []
    for count, model in enumerate(results.index.to_frame()['model'].unique()):
        mask_for_model = results.index.get_level_values('model') == model
        data = results.loc[mask_for_dataset & mask_for_model & mask_for_transform]
        model_label = f"{model} ({transform})" if transform != "no_transform" else str(model)
        entries.append(DataEntry(model_label, data, count % 2 == 0))

    plot_def = PlotDef(title=f"{dataset}, {transform}", entries=entries, legend=legend)
    return errorbox(plot, plot_def, xaxis, yaxis, 0, 0, markersize, use_cross)


def plot_mean_std_box(
    results: pd.DataFrame,
    metric_y: Union[str, Metric],
    metric_x: Union[str, Metric],
    save: bool = True,
    dpi: int = 300,
    use_cross: bool = False,
) -> List[Tuple[plt.Figure, plt.Axes]]:
    """Plot the given result with boxes that represent mean and standard deviation

    Args:
        results: a DataFrame that already contains the values of the metrics
        metric_y: a Metric object or a column name that defines which metric to plot on the y-axis
        metric_x: a Metric object or a column name that defines which metric to plot on the x-axis
        save: if True, save the plot as a PDF
        dpi: DPI of the plots
    Returns:
        A list of all figures and plots
    """
    directory = Path(".") / "plots"
    directory.mkdir(exist_ok=True)

    def _get_columns(metric: Metric) -> List[str]:
        cols = [col for col in results.columns if metric.name in col]
        if not cols:
            raise ValueError(f"No matching columns found for Metric \"{metric.name}\".")
        # if there are multiple matches, then the metric was `per_sensitive_attribute`. In this
        # case, we *only* want ratios and differences; not the plain result
        if len(cols) > 1:
            cols = [col for col in cols if ("-" in col) or ("/" in col)]
        return cols

    if isinstance(metric_x, Metric):
        # if the metric is given as a Metric object, look for matching columns
        cols_x = _get_columns(metric_x)
    else:
        if metric_x not in results.columns:
            raise ValueError(f"No column named \"{metric_x}\".")
        cols_x = [metric_x]

    if isinstance(metric_y, Metric):
        cols_y = _get_columns(metric_y)
    else:
        if metric_y not in results.columns:
            raise ValueError(f"No column named \"{metric_y}\".")
        cols_y = [metric_y]

    # generate the Cartesian product of `cols_x` and `cols_y`; i.e. all possible combinations
    # this preserves the order of x and y
    possible_pairs = list(itertools.product(cols_x, cols_y))

    figure_list: List[Tuple[plt.Figure, plt.Axes]] = []
    for dataset in results.index.to_frame()['dataset'].unique():
        dataset_: str = str(dataset)
        for transform in results.index.to_frame()['transform'].unique():
            transform_: str = str(transform)
            for x_axis, y_axis in possible_pairs:
                fig: plt.Figure
                plot: plt.Axes
                fig, plot = plt.subplots(dpi=dpi)

                xtuple = (x_axis, x_axis.replace("_", " "))
                ytuple = (y_axis, y_axis.replace("_", " "))
                legend = single_plot_mean_std_box(
                    plot, results, xtuple, ytuple, dataset_, transform_, use_cross=use_cross
                )

                if legend is False:  # use "is" here because otherwise any falsy value would match
                    plt.close(fig)
                    continue  # nothing was plotted -> don't save it and don't add it to the list

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
