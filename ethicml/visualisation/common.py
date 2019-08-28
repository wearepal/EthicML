"""
common plotting functions / datastructures
"""

from typing import Tuple, Union, NamedTuple, List

import pandas as pd
import matplotlib.pyplot as plt


class DataEntry(NamedTuple):
    """Defines one element in a plot"""

    label: str
    values: pd.DataFrame
    do_fill: bool


class PlotDef(NamedTuple):
    title: str
    entries: List[DataEntry]


def common_plotting_settings(
    plot: plt.Axes,
    plot_def: PlotDef,
    xaxis_title: str,
    yaxis_title: str,
    legend: Union[str, Tuple[str, float]] = "inside",
):
    """Common settings for plots
    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis_title: label for x-axis
        yaxis_title: label for y-axis
        legend: where to put the legend; allowed values: None, "inside", "outside"
    """
    plot.set_xlabel(xaxis_title)
    plot.set_ylabel(yaxis_title)
    if plot_def.title:
        plot.set_title(plot_def.title)
    plot.grid(True)
    legend_ = plot.legend()
    if legend == "outside":
        legend_ = plot.legend(loc='upper left', bbox_to_anchor=(1, 1))
    elif isinstance(legend, tuple) and legend[0] == "outside" and isinstance(legend[1], float):
        legend_ = plot.legend(bbox_to_anchor=(1, legend[1]), loc=2)
    return legend_


def errorbox(
    plot: plt.Axes,
    plot_def: PlotDef,
    xaxis: Tuple[str, str],
    yaxis: Tuple[str, str],
    legend: Union[str, Tuple[str, float]] = "inside",
    firstcolor: int = 0,
    firstshape: int = 0,
    markersize: int = 6,
    use_cross: bool = False,
):
    """Generate a figure with errorboxes that reflect the std dev of an entry
    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        legend: where to put the legend; allowed values: None, "inside", "outside"
    """
    # ================================ constants for plots ========================================
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    colors += ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    pale_colors = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]
    pale_colors += ["#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]
    shapes = ['o', 'X', 'D', 's', '^', 'v', '<', '>', '*', 'p', 'P']
    hatch_pattern = ['O', 'o', '|', '-', '/', '\\', '+', 'x', '*']

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
                xerr=xstd,
                yerr=ystd,
                fmt=shapes[i_shp % len(shapes)],
                color=color,
                ecolor='k',
                capsize=4,
                elinewidth=2.0,
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
                align='center',
                color='none',
                edgecolor=color,
                linewidth=3,
                zorder=3 + 2 * i_shp,
                hatch=hatch_pattern[i % len(hatch_pattern)],
            )
            plot.plot(
                xmean,
                ymean,
                shapes[i_shp % len(shapes)],
                c=color,
                label=entry.label,
                zorder=4 + 2 * i_shp,
                markersize=markersize,
            )

    x_min = min_xmean * 0.99
    x_max = max_xmean * 1.01

    y_min = min_ymean * 0.99
    y_max = max_ymean * 1.01

    plot.set_xlim([x_min, x_max])
    plot.set_ylim([y_min, y_max])
    return common_plotting_settings(plot, plot_def, xaxis[1], yaxis[1], legend)
