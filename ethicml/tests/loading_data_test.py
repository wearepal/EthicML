"""
Test the loading data capability
"""

from typing import Dict
import pandas as pd

from ethicml.common import ROOT_DIR
from ethicml.data.test import Test
from ethicml.data.load import load_data


def test_can_load_test_data():
    data_loc: str = "{}/data/csvs/test.csv".format(ROOT_DIR)
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


def test_load_data():
    data: Dict[str, pd.DataFrame] = load_data(Test())
    assert (2000, 2) == data['x'].shape
    assert (2000, 1) == data['s'].shape
    assert (2000, 1) == data['y'].shape
