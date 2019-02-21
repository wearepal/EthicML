"""
creates plots of a dataset
"""


import os
from pathlib import Path
from typing import Dict, Tuple, List

import imageio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MARKERS = ['s', 'p', 'P', '*', '+', 'x', 'o', 'v']


def save_2d_plot(data: Dict[str, pd.DataFrame], filepath: str):
    """

    Args:
        data:
        filepath:
    """
    file_path = Path(filepath)
    columns = data['x'].columns

    amalgamated = pd.concat([data['x'], data['s'], data['y']], axis=1)

    plot = sns.scatterplot(x=columns[0], y=columns[1], hue=data['y'].columns[0], palette="Set2",
                           data=amalgamated, style=data['s'].columns[0], legend='full')

    if not os.path.exists(file_path.parent):
        os.mkdir(file_path. parent)
    plot.figure.savefig(file_path)
    plt.clf()


def save_jointplot(data: Dict[str, pd.DataFrame], filepath: str, dims: Tuple[int, int] = (0, 1)):
    """

    Args:
        data:
        filepath:
        dims:
    """
    file_path = Path(filepath)
    columns = data['x'].columns

    amalgamated = pd.concat([data['x'], data['y']], axis=1)

    plot = sns.jointplot(x=columns[dims[0]], y=columns[dims[1]], data=amalgamated, kind='kde')

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
    imageio.mimsave(f'results/{name}.gif', images)
