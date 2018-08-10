"""
Test preprocessing capabilities
"""

from typing import Tuple
import pandas as pd
import pytest
from ethicml.data.load import load_data
from ethicml.preprocessing.train_test_split import train_test_split


def test_train_test_split():
    data: pd.DataFrame = load_data('test')
    train_test: Tuple[pd.DataFrame, pd.DataFrame] = train_test_split(data)
    train, test = train_test
    assert pytest.approx(train['a1'].iloc[0], -0.466241)
    assert pytest.approx(test['a1'].iloc[0], -0.466241)
    assert train is not None
    assert test is not None
    assert not train['a1'].iloc[0] == 1.4410847587041449
