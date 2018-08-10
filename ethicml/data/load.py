"""
Loads Data from .csv files
"""

import pandas as pd

from ethicml.common import ROOT_DIR


def load_data(name: str) -> pd.DataFrame:
    data_loc: str = "{}/data/csvs/{}.csv".format(ROOT_DIR, name)
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert isinstance(data, pd.DataFrame)
    return data
