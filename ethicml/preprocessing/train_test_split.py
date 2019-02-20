"""
Split into train and test data
"""
from typing import Tuple, Dict, List
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd


def call_numpy_to_split(dataframe: pd.DataFrame, train_percentage, random_seed) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_test: Tuple[pd.DataFrame, pd.DataFrame] = \
        np.split(dataframe.sample(frac=1, random_state=random_seed).reset_index(drop=True),
                 [int(train_percentage * len(dataframe))])

    assert isinstance(train_test[0], pd.DataFrame)
    assert isinstance(train_test[1], pd.DataFrame)

    train = train_test[0].reset_index(drop=True)
    test = train_test[1].reset_index(drop=True)

    return train, test


def train_test_split(data: Dict[str, pd.DataFrame],
                     train_percentage: float = 0.8,
                     random_seed: int = 0) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    x_columns: List[str] = [col for col in data['x'].columns]
    s_columns: List[str] = [col for col in data['s'].columns]
    y_columns: List[str] = [col for col in data['y'].columns]
    # ty_columns: List[str] = [col for col in data['ty'].columns]

    all_data: pd.DataFrame = pd.concat([data['x'], data['s'], data['y']],#, data['ty']],
                                       axis=1)

    all_data = all_data.sample(frac=1, random_state=1).reset_index(drop=True)

    all_data_train_test: Tuple[pd.DataFrame, pd.DataFrame] = call_numpy_to_split(all_data,
                                                                                 train_percentage,
                                                                                 random_seed=random_seed)

    all_data_train, all_data_test = all_data_train_test

    train: Dict[str, pd.DataFrame] = {
        'x': all_data_train[x_columns],
        's': all_data_train[s_columns],
        'y': all_data_train[y_columns],
        # 'ty': all_data_train[ty_columns]
    }

    test: Dict[str, pd.DataFrame] = {
        'x': all_data_test[x_columns],
        's': all_data_test[s_columns],
        'y': all_data_test[y_columns],
        # 'ty': all_data_test[ty_columns]
    }

    assert isinstance(train['x'], pd.DataFrame)
    assert isinstance(test['x'], pd.DataFrame)
    assert_array_equal(train['x'].columns, x_columns)
    assert_array_equal(test['x'].columns, x_columns)

    assert isinstance(train['s'], pd.DataFrame)
    assert isinstance(test['s'], pd.DataFrame)
    assert_array_equal(train['s'].columns, s_columns)
    assert_array_equal(test['s'].columns, s_columns)

    assert isinstance(train['y'], pd.DataFrame)
    assert isinstance(test['y'], pd.DataFrame)
    assert_array_equal(train['y'].columns, y_columns)
    assert_array_equal(test['y'].columns, y_columns)

    return train, test
