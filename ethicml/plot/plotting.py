"""Create plots of a dataset."""
from __future__ import annotations
import itertools
from pathlib import Path
from typing import Any, Callable, Literal

from matplotlib import figure, legend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ethicml.metrics.metric import Metric
from ethicml.utility import DataTuple, Results

from .common import DataEntry, LegendType, PlotDef, PlotType, errorbox, scatter

__all__ = [
    "plot_results",
    "single_plot",
    "save_2d_plot",
    "save_label_plot",
    "save_jointplot",
    "save_multijointplot",
]

MARKERS = ["s", "p", "P", "*", "+", "x", "o", "v"]


def maybe_tsne(data: DataTuple) -> tuple[pd.DataFrame, str, str]:
    columns = data.x.columns

    if len(columns) > 2:
        from sklearn.manifold import TSNE  # type: ignore[attr-defined]

        tsne_embeddings = TSNE(n_components=2, random_state=0).fit_transform(data.x)
        amalgamated = pd.concat(
            [pd.DataFrame(tsne_embeddings, columns=["tsne1", "tsne2"]), data.s, data.y],
            axis="columns",
        )
        x1_name = "tsne1"
        x2_name = "tsne2"
    else:
        amalgamated = data.data
        x1_name = f"{columns[0]}"  # type: ignore[str-bytes-safe]
        x2_name = f"{columns[1]}"  # type: ignore[str-bytes-safe]
    return amalgamated, x1_name, x2_name


def save_2d_plot(data: DataTuple, filepath: str) -> None:
    """Make 2D plot."""
    file_path = Path(filepath)

    amalgamated, x1_name, x2_name = maybe_tsne(data)

    plot = sns.scatterplot(  # type: ignore[attr-defined]
        x=x1_name,
        y=x2_name,
        hue=data.y.name,
        palette="Set2",
        data=amalgamated,
        style=data.s.name,
        legend="full",
    )

    file_path.parent.mkdir(exist_ok=True)
    plot.figure.savefig(file_path)
    plt.clf()


def save_jointplot(data: DataTuple, filepath: str, dims: tuple[int, int] = (0, 1)) -> None:
    """Make joint plot."""
    file_path = Path(filepath)
    columns = data.x.columns

    amalgamated = pd.concat([data.x, data.y], axis="columns")

    plot = sns.jointplot(x=columns[dims[0]], y=columns[dims[1]], data=amalgamated, kind="kde")  # type: ignore[attr-defined]

    file_path.parent.mkdir(exist_ok=True)
    plot.savefig(file_path)
    plt.clf()


def multivariate_grid(
    col_x: str,
    col_y: str,
    sens_col: str,
    outcome_col: str,
    df: pd.DataFrame,
    scatter_alpha: float = 0.5,
) -> None:
    def colored_scatter(x: Any, y: Any, c: str | None = None) -> Callable[[Any], None]:
        def scatter(*args: Any, **kwargs: Any) -> None:
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    sns.set_palette("husl")  # type: ignore[attr-defined]

    g = sns.JointGrid(x=col_x, y=col_y, data=df)  # type: ignore[attr-defined]
    color = None
    legends = []
    for name, df_group in df.groupby([sens_col, outcome_col]):
        legends.append(f"S={name[0]}, Y={name[1]}")
        g.plot_joint(colored_scatter(df_group[col_x], df_group[col_y], color))
        sns.distplot(df_group[col_x].values, ax=g.ax_marg_x, color=color)  # type: ignore[attr-defined]
        sns.distplot(df_group[col_y].values, ax=g.ax_marg_y, vertical=True)  # type: ignore[attr-defined]
    # Do also global Hist:
    # sns.distplot(df[col_x].values, ax=g.ax_marg_x, color='grey')
    # sns.distplot(df[col_y].values.ravel(), ax=g.ax_marg_y, color='grey', vertical=True)
    plt.legend(legends)


def save_multijointplot(data: DataTuple, filepath: str) -> None:
    """Make joint plot."""
    file_path = Path(filepath)
    data.x.columns

    amalgamated, x1_name, x2_name = maybe_tsne(data)

    multivariate_grid(
        col_x=x1_name,
        col_y=x2_name,
        sens_col=str(data.s.name),
        outcome_col=str(data.y.name),
        df=amalgamated,
    )

    file_path.parent.mkdir(exist_ok=True)
    plt.savefig(file_path)
    plt.clf()


def make_gif(files: list[str], name: str = "movie") -> None:
    """Make GIF."""
    import imageio

    images = [imageio.imread(filename) for filename in files]
    imageio.mimsave(f"{name}.gif", images)


def save_label_plot(data: DataTuple, filename: str) -> None:
    """Make label plot."""
    file_path = Path(filename)

    # Only consider 1 sens attr for now
    s_values = data.s.value_counts() / data.s.count()

    s_0_val = s_values[0]
    s_1_val = s_values[1]

    s_0_label = s_values.index.min()
    s_1_label = s_values.index.max()

    y_s0 = data.y[data.s == 0].value_counts() / data.y[data.s == 0].count()
    y_s1 = data.y[data.s == 1].value_counts() / data.y[data.s == 1].count()

    y_0_label = y_s0.index[0]
    y_1_label = y_s0.index[1]

    plt.style.use("seaborn-pastel")
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
        height=y_s1[y_0_label] * 100,  # type: ignore[index]
        width=s_1_val * 100,
        align="edge",
        edgecolor="black",
        color="C1",
    )
    quadrant3 = plot.bar(
        0,
        height=y_s0[y_1_label] * 100,  # type: ignore[index]
        width=s_0_val * 100,
        bottom=y_s0[y_0_label] * 100,  # type: ignore[index]
        align="edge",
        edgecolor="black",
        color="C2",
    )
    quadrant4 = plot.bar(
        s_0_val * 100,
        height=y_s1[y_1_label] * 100,  # type: ignore[index]
        width=s_1_val * 100,
        bottom=y_s1[y_0_label] * 100,  # type: ignore[index]
        align="edge",
        edgecolor="black",
        color="C3",
    )

    plot.set_ylim(0, 100)
    plot.set_xlim(0, 100)
    plot.set_ylabel(f"Percent {data.y.name}=y")
    plot.set_xlabel(f"Percent {data.s.name}=s")
    plot.set_title("Dataset Composition by class and sensitive attribute")

    plot.legend(
        [quadrant1, quadrant2, quadrant3, quadrant4],
        [
            f"y={y_0_label}, s={s_0_label}",  # type: ignore[str-bytes-safe]
            f"y={y_0_label}, s={s_1_label}",  # type: ignore[str-bytes-safe]
            f"y={y_1_label}, s={s_0_label}",  # type: ignore[str-bytes-safe]
            f"y={y_1_label}, s={s_1_label}",  # type: ignore[str-bytes-safe]
        ],
    )

    file_path.parent.mkdir(exist_ok=True)
    fig.savefig(file_path)
    fig.clf()


def single_plot(
    plot: plt.Axes,
    results: Results,
    xaxis: tuple[str, str],
    yaxis: tuple[str, str],
    dataset: str,
    transform: str | None,
    ptype: PlotType = "box",
    legend_pos: LegendType | None = "outside",
    legend_yanchor: float = 1.0,
    markersize: int = 6,
    alternating_style: bool = True,
    include_nan_entries: bool = False,
) -> None | Literal[False] | legend.Legend:
    """Provide the functionality of the individual plotting functions through a nice interface.

    This function can also be used to create figures with multiple plots on them, because it does
    not generate a Figure object itself.

    :param plot: Plot object
    :param results: DataFrame with the data
    :param xaxis: Name of column that's plotted on the x-axis.
    :param yaxis: Name of column that's plotted on the y-axis.
    :param dataset: string that identifies the dataset
    :param transform: string that identifies the preprocessing method, or None
    :param ptype: plot type (Default: "box")
    :param legend_pos: position of the legend (or None for no legend) (Default: "outside")
    :param legend_yanchor: position in the vertical direction where the legend should begin (Default
        value = 1.0)
    :param markersize: size of marker (Default: 6)
    :param alternating_style: If True, entries for scatter plots are done in alternating style
        (Default: True)
    :param include_nan_entries: If True, entries with NaNs still appear in the legend (Default value
        = False)
    :returns: The legend object if something was plotted; False otherwise.
    :raises ValueError: If ``ptype`` has an unsupported value.
    """
    mask_for_dataset = results.index.get_level_values("dataset") == dataset
    if transform is not None:
        transforms: list[str] = [transform]
    else:
        transforms = [str(t) for t in results.index.to_frame()["transform"].unique()]

    entries: list[DataEntry] = []
    count = 0
    for model in results.index.to_frame()["model"].unique():
        mask_for_model = results.index.get_level_values("model") == model
        for transform_ in transforms:
            mask_for_transform = results.index.get_level_values("transform") == transform_
            data = results.loc[mask_for_dataset & mask_for_model & mask_for_transform]
            if (
                data[[xaxis[0], yaxis[0]]].empty or data[[xaxis[0], yaxis[0]]].isnull().any().any()
            ) and not include_nan_entries:
                continue  # this entry has missing values
            model_label = f"{model} ({transform_})" if transform_ != "no_transform" else str(model)
            entries.append(DataEntry(model_label, data, (not alternating_style) or count % 2 == 0))
            count += 1

    if not entries:
        return False  # nothing to plot

    title = f"{dataset}, {transform}" if transform is not None else dataset
    plot_def = PlotDef(
        title=title, entries=entries, legend_pos=legend_pos, legend_yanchor=legend_yanchor
    )
    if ptype == "box":
        return errorbox(plot, plot_def, xaxis, yaxis, 0, 0, markersize, use_cross=False)
    if ptype == "cross":
        return errorbox(plot, plot_def, xaxis, yaxis, 0, 0, markersize, use_cross=True)
    if ptype == "scatter":
        return scatter(plot, plot_def, xaxis, yaxis, 0, markersize, connect_dots=False)
    if ptype == "line":
        return scatter(plot, plot_def, xaxis, yaxis, 0, markersize, connect_dots=True)
    raise ValueError(f"Unknown plot type '{ptype}'")


def plot_results(
    results: Results,
    metric_y: str | Metric,
    metric_x: str | Metric,
    ptype: PlotType = "box",
    save: bool = True,
    dpi: int = 300,
    transforms_separately: bool = True,
) -> list[tuple[figure.Figure, plt.Axes]]:
    """Plot the given result with boxes that represent mean and standard deviation.

    :param results: a DataFrame that already contains the values of the metrics
    :param metric_y: a Metric object or a column name that defines which metric to plot on the y-axis
    :param metric_x: a Metric object or a column name that defines which metric to plot on the x-axis
    :param ptype: plot type (Default: "box")
    :param save: if True, save the plot as a PDF (Default: True)
    :param dpi: DPI of the plots (Default: 300)
    :param transforms_separately: if True, each transform gets its own plot (Default: True)
    :returns: A list of all figures and plots
    :raises ValueError: If no columns matching ``metric_y`` and ``metric_x`` are found.
    """
    directory = Path(".") / "plots"
    directory.mkdir(exist_ok=True)

    def _get_columns(metric: Metric) -> list[str]:
        cols = [col for col in results.columns if metric.name in col]
        if not cols:
            raise ValueError(f'No matching columns found for Metric "{metric.name}".')
        # if there are multiple matches, then the metric was `per_sensitive_attribute`. In this
        # case, we *only* want ratios and differences; not the plain result
        if len(cols) > 1:
            cols = [col for col in cols if ("-" in col) or ("÷" in col)]
        return cols

    if isinstance(metric_x, str):
        if metric_x not in results.columns:
            raise ValueError(f'No column named "{metric_x}".')
        cols_x = [metric_x]
    else:
        # if the metric is given as a Metric object, look for matching columns
        cols_x = _get_columns(metric_x)

    if isinstance(metric_y, str):
        if metric_y not in results.columns:
            raise ValueError(f'No column named "{metric_y}".')
        cols_y = [metric_y]
    else:
        cols_y = _get_columns(metric_y)

    # generate the Cartesian product of `cols_x` and `cols_y`; i.e. all possible combinations
    # this preserves the order of x and y
    possible_pairs = list(itertools.product(cols_x, cols_y))

    transforms: list[str | None]
    if transforms_separately:
        transforms = [str(t) for t in results.index.to_frame()["transform"].unique()]
    else:
        transforms = [None]

    figure_list: list[tuple[figure.Figure, plt.Axes]] = []
    for dataset in results.index.to_frame()["dataset"].unique():
        dataset_: str = str(dataset)
        for transform in transforms:
            for x_axis, y_axis in possible_pairs:
                fig: figure.Figure
                plot: plt.Axes
                fig, plot = plt.subplots(dpi=dpi)

                xtuple = (x_axis, x_axis.replace("_", " "))
                ytuple = (y_axis, y_axis.replace("_", " "))
                legend = single_plot(
                    plot, results, xtuple, ytuple, dataset_, transform, ptype=ptype
                )
                plt.tight_layout()

                if legend is False:  # use "is" here because otherwise any falsy value would match
                    plt.close(fig)
                    continue  # nothing was plotted -> don't save it and don't add it to the list

                if save:
                    fig.savefig(
                        directory / f"{dataset} {transform} {x_axis} {y_axis}.pdf",
                        bbox_inches="tight",
                    )
                plt.close(fig)
                figure_list += [(fig, plot)]
    return figure_list
