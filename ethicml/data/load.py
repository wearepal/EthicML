"""
Loads Data from .csv files
"""

import pandas as pd
from ethicml.common import ROOT_DIR
from ethicml.data.dataset import Dataset


def load_data(dataset: Dataset) -> dict:
    filename = dataset.get_filename()
    data_loc: str = "{}/data/csvs/{}.csv".format(ROOT_DIR, filename)
    dataframe: pd.DataFrame = pd.read_csv(data_loc)
    assert isinstance(dataframe, pd.DataFrame)

    feature_split = dataset.get_feature_split()

    x_data = dataframe[feature_split['x']]
    s_data = dataframe[feature_split['s']]
    y_data = dataframe[feature_split['y']]

    return {"x": x_data,
            "s": s_data,
            "y": y_data}
