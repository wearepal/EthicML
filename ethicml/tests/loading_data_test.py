"""
Test the loading data capability
"""

from typing import Dict
import pandas as pd

from ethicml.common import ROOT_DIR
from ethicml.data.test import Test
from ethicml.data.load import load_data, create_data_obj
from ethicml.data.dataset import Dataset


def test_can_load_test_data():
    data_loc: str = "{}/data/csvs/test.csv".format(ROOT_DIR)
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


def test_load_data():
    data: Dict[str, pd.DataFrame] = load_data(Test())
    assert (2000, 2) == data['x'].shape
    assert (2000, 1) == data['s'].shape
    assert (2000, 1) == data['y'].shape


def test_load_data_as_a_function():
    data_loc: str = "{}/data/csvs/test.csv".format(ROOT_DIR)
    data_obj: Dataset = create_data_obj(data_loc, s_columns=["s"], y_columns=["y"])
    assert data_obj is not None
    assert data_obj.get_feature_split()['x'] == ['a1', 'a2']
    assert data_obj.get_feature_split()['s'] == ['s']
    assert data_obj.get_feature_split()['y'] == ['y']
    assert data_obj.get_filename() == "test.csv"

def test_joing_2_load_function():
    data_loc: str = "{}/data/csvs/test.csv".format(ROOT_DIR)
    data_obj: Dataset = create_data_obj(data_loc, s_columns=["s"], y_columns=["y"])
    data: Dict[str, pd.DataFrame] = load_data(data_obj)
    assert (2000, 2) == data['x'].shape
    assert (2000, 1) == data['s'].shape
    assert (2000, 1) == data['y'].shape
