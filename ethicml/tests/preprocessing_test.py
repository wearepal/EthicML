"""
Test preprocessing capabilities
"""
import math

from typing import Tuple, Dict
import pandas as pd

from ethicml.data.test import Test
from ethicml.data.load import load_data
from ethicml.preprocessing.train_test_split import train_test_split


def test_train_test_split():
    data: Dict[str, pd.DataFrame] = load_data(Test())
    train_test: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]] = train_test_split(data)
    train, test = train_test
    assert train is not None
    assert test is not None
    assert train['x'].shape[0] > test['x'].shape[0]
    assert train['x']['a1'].iloc[0] == 0.6431486026206228
    assert train['x']['a2'].iloc[0] == -0.09879806963941018

    assert train['x'].shape[0] == train['s'].shape[0]
    assert train['s'].shape[0] == train['y'].shape[0]

    len_default = math.floor((2000/100)*80)
    assert train['s'].shape[0] == len_default
    assert test['s'].shape[0] == 2000 - len_default

    len_0_9 = math.floor((2000/100)*90)
    train, test = train_test_split(data, train_percentage=0.9)
    assert train['s'].shape[0] == len_0_9
    assert test['s'].shape[0] == 2000 - len_0_9

    len_0_7 = math.floor((2000 / 100) * 70)
    train, test = train_test_split(data, train_percentage=0.7)
    assert train['s'].shape[0] == len_0_7
    assert test['s'].shape[0] == 2000 - len_0_7

    len_0_5 = math.floor((2000 / 100) * 50)
    train, test = train_test_split(data, train_percentage=0.5)
    assert train['s'].shape[0] == len_0_5
    assert test['s'].shape[0] == 2000 - len_0_5

    len_0_3 = math.floor((2000 / 100) * 30)
    train, test = train_test_split(data, train_percentage=0.3)
    assert train['s'].shape[0] == len_0_3
    assert test['s'].shape[0] == 2000 - len_0_3

    len_0_1 = math.floor((2000 / 100) * 10)
    train, test = train_test_split(data, train_percentage=0.1)
    assert train['s'].shape[0] == len_0_1
    assert test['s'].shape[0] == 2000 - len_0_1

    len_0_0 = math.floor((2000 / 100) * 0)
    train, test = train_test_split(data, train_percentage=0.0)
    assert train['s'].shape[0] == len_0_0
    assert test['s'].shape[0] == 2000 - len_0_0
