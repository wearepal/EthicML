"""
Test the loading data capability
"""

import pandas as pd

from ethicml.common import ROOT_DIR
from ethicml.data.load import load_data


def test_can_load_test_data():
    data_loc: str = "{}/data/csvs/test.csv".format(ROOT_DIR)
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


def test_load_data():
    data: pd.DataFrame = load_data('test')
    assert data.shape == (200, 3)
