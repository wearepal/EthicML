"""Common plotting functions / datastructures."""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from typing_extensions import Literal

LegendType = Literal["inside", "outside"]  # pylint: disable=invalid-name
PlotType = Literal["box", "cross", "scatter", "line"]  # pylint: disable=invalid-name


class DataEntry(NamedTuple):
    """Defines one element in a plot."""

    label: str
    values: pd.DataFrame
    do_fill: bool


class PlotDef(NamedTuple):
    """All the things needed to make a plot.

    Attributes:
        title: the title of the plot
        entries: a List of DataEntries
        legend_pos: where to put the legend; allowed values: None, "inside", "outside"
        legend_yanchor: float that specifies the vertical position of the legend
    """

    title: str
    entries: List[DataEntry]
    legend_pos: Optional[LegendType] = None
    legend_yanchor: float = 1.0


def common_plotting_settings(
    plot: plt.Axes, plot_def: PlotDef, xaxis_title: str, yaxis_title: str
) -> Optional[mpl.legend.Legend]:
    """Common settings for plots.

    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis_title: label for x-axis
        yaxis_title: label for y-axis
    """
    plot.set_xlabel(xaxis_title)
    plot.set_ylabel(yaxis_title)
    if plot_def.title:
        plot.set_title(plot_def.title)
    plot.grid(True)

    # get handles and labels for the legend
    handles, labels = plot.get_legend_handles_labels()
    # remove the errorbars from the legend if they are there
    handles = [(h[0] if isinstance(h, tuple) else h) for h in handles]

    if plot_def.legend_pos == "inside":
        return plot.legend(handles, labels)
    if plot_def.legend_pos == "outside":
        return plot.legend(
            handles, labels, loc="upper left", bbox_to_anchor=(1, plot_def.legend_yanchor)
        )
    return None


def errorbox(
    plot: plt.Axes,
    plot_def: PlotDef,
    xaxis: Tuple[str, str],
    yaxis: Tuple[str, str],
    firstcolor: int = 0,
    firstshape: int = 0,
    markersize: int = 6,
    use_cross: bool = False,
) -> Optional[mpl.legend.Legend]:
    """Generate a figure with errorboxes that reflect the std dev of an entry.

    Args:
        plot: a pyplot Axes object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis: a tuple of two strings
        yaxis: a tuple of two strings
        firstcolor: index of the color that should be used for the first entry
        firstshape: index of the shape that should be used for the first entry
        markersize: size of the markers
        use_cross: if True, use cross instead of boxes
    """
    # ================================ constants for plots ========================================
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    colors += ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    pale_colors = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]
    pale_colors += ["#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]
    shapes = ["o", "X", "D", "s", "^", "v", "<", ">", "*", "p", "P"]
    hatch_pattern = ["O", "o", "|", "-", "/", "\\", "+", "x", "*"]

    xaxis_measure, yaxis_measure = xaxis[0], yaxis[0]
    filled_counter = firstcolor

    max_xmean = 0.0
    min_xmean = 1.0

    max_ymean = 0.0
    min_ymean = 1.0

    for i, entry in enumerate(plot_def.entries):
        if entry.do_fill:
            color = colors[filled_counter]
            filled_counter += 1
        else:
            color = pale_colors[i + firstcolor - filled_counter]
        i_shp = firstshape + i

        xmean: float = entry.values[xaxis_measure].mean()
        xstd: float = entry.values[xaxis_measure].std()
        if xmean + (0.5 * xstd) > max_xmean:
            max_xmean = xmean + (0.5 * xstd)
        if xmean - (0.5 * xstd) < min_xmean:
            min_xmean = xmean - (0.5 * xstd)

        ymean: float = entry.values[yaxis_measure].mean()
        ystd: float = entry.values[yaxis_measure].std()
        if ymean + (0.5 * ystd) > max_ymean:
            max_ymean = ymean + (0.5 * ystd)
        if ymean - (0.5 * ystd) < min_ymean:
            min_ymean = ymean - (0.5 * ystd)

        if use_cross:
            plot.errorbar(
                xmean,
                ymean,
                xerr=0.5 * xstd,
                yerr=0.5 * ystd,
                marker=shapes[i_shp % len(shapes)],
                linestyle="",  # no connecting lines
                color=color,
                ecolor="#555555",
                capsize=4,
                elinewidth=1.0,
                zorder=3 + 2 * i_shp,
                label=entry.label,
                markersize=markersize,
            )
        else:
            plot.bar(
                xmean,
                ystd,
                bottom=ymean - 0.5 * ystd,
                width=xstd,
                align="center",
                color="none",
                edgecolor=color,
                linewidth=3,
                zorder=3 + 2 * i_shp,
                hatch=hatch_pattern[i % len(hatch_pattern)],
            )
            plot.plot(
                xmean,
                ymean,
                marker=shapes[i_shp % len(shapes)],
                linestyle="",  # no connecting lines
                color=color,
                label=entry.label,
                zorder=4 + 2 * i_shp,
                markersize=markersize,
            )

    x_min = min_xmean * 0.99
    x_max = max_xmean * 1.01

    y_min = min_ymean * 0.99
    y_max = max_ymean * 1.01

    plot.set_xlim(x_min, x_max)
    plot.set_ylim(y_min, y_max)
    return common_plotting_settings(plot, plot_def, xaxis[1], yaxis[1])


def scatter(
    plot: plt.Axes,
    plot_def: PlotDef,
    xaxis: Tuple[str, str],
    yaxis: Tuple[str, str],
    startindex: int = 0,
    markersize: int = 6,
    connect_dots: bool = False,
) -> Optional[mpl.legend.Legend]:
    """Generate a scatter plot.

    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis: a tuple of two strings
        yaxis: a tuple of two strings
        startindex: index of color and shape that should be used for the first entry
        markersize: size of the markers
        connect_dots: if True, connect the data points with a line
    """
    shapes = ["o", "X", "D", "s", "^", "v", "<", ">", "*", "p", "P"]
    colors10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    colors10 += ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    xaxis_measure, yaxis_measure = xaxis[0], yaxis[0]
    filled_counter = startindex
    for i, entry in enumerate(plot_def.entries):
        additional_params: Dict[str, Any]
        if entry.do_fill or connect_dots:
            additional_params = {}
            shp_index = filled_counter
            filled_counter += 1
        else:
            additional_params = dict(markerfacecolor="none")
            shp_index = i + startindex - filled_counter
        plot.plot(
            entry.values[xaxis_measure].to_numpy(),
            entry.values[yaxis_measure].to_numpy(),
            marker=shapes[shp_index],
            linestyle="-" if connect_dots else "",
            label=entry.label,
            color=None if connect_dots else colors10[shp_index],
            markersize=markersize,
            **additional_params,
        )
    return common_plotting_settings(plot, plot_def, xaxis[1], yaxis[1])
