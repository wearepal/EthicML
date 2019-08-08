"""
common plotting functions / datastructures
"""

from collections import namedtuple
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

DataEntry = namedtuple('DataEntry', ['label', 'values', 'do_fill'])
PlotDef = namedtuple('PlotDef', ['title', 'entries'])


def common_plotting_settings(plot, plot_def, xaxis_title, yaxis_title, legend: str = "inside"):
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
    plot: plt.plot,
    plot_def: PlotDef,
    xaxis: Tuple[str, str],
    yaxis: Tuple[str, str],
    legend: str = "inside",
    firstcolor: int = 0,
    firstshape: int = 0,
    markersize: int = 6,
):
    """Generate a figure with errorboxes that reflect the std dev of an entry
    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        legend: where to put the legend; allowed values: None, "inside", "outside"
    """

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    pale_colors = [
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]
    shapes = ['o', 'X', 'D', 's', '^', 'v', '<', '>', '*', 'p', 'P']

    hatch_pattern = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    xaxis_measure, yaxis_measure = xaxis[0], yaxis[0]
    filled_counter = firstcolor

    max_xmean: float = 0.0
    min_xmean: float = 1.0

    max_ymean: float = 0.0
    min_ymean: float = 1.0

    for i, entry in enumerate(plot_def.entries):
        if entry.do_fill:
            color = colors[filled_counter]
            filled_counter += 1
        else:
            color = pale_colors[i + firstcolor - filled_counter]
        i_shp = firstshape + i

        xmean: float = float(np.mean(entry.values[xaxis_measure]))
        xstd: float = float(np.std(entry.values[xaxis_measure]))
        if xmean + (0.5 * xstd) > max_xmean:
            max_xmean = xmean + (0.5 * xstd)
        if xmean - (0.5 * xstd) < min_xmean:
            min_xmean = xmean - (0.5 * xstd)

        ymean: float = float(np.mean(entry.values[yaxis_measure]))
        ystd: float = float(np.std(entry.values[yaxis_measure]))
        if ymean + (0.5 * ystd) > max_ymean:
            max_ymean = ymean + (0.5 * ystd)
        if ymean - (0.5 * ystd) < min_ymean:
            min_ymean = ymean - (0.5 * ystd)

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
            shapes[i_shp],
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
