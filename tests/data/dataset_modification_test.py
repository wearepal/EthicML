"""Test modifiactions to a dataset."""
from typing import Type, Union

import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ethicml import scale_continuous, train_test_split
from ethicml.data import available_tabular, get_dataset_obj_by_name
from ethicml.data.util import from_dummies


@pytest.mark.parametrize("dataset_name", available_tabular())
@pytest.mark.parametrize("scaler_type", [StandardScaler, MinMaxScaler])
def test_scaling(
    dataset_name: str, scaler_type: Union[Type[StandardScaler], Type[MinMaxScaler]]
) -> None:
    """Test that scaling works."""
    scaler = scaler_type()
    dataset = get_dataset_obj_by_name(dataset_name)()
    datatuple = dataset.load()

    # Speed up the tests by making some data smaller
    if dataset_name == "health":
        datatuple, _ = train_test_split(datatuple, train_percentage=0.05)

    datatuple_scaled, scaler2 = scale_continuous(dataset, datatuple, scaler)

    if dataset_name == "crime" and str(scaler) == "MinMaxScaler()":
        # Crime dataset is minmax scaled by the data providers.
        pd.testing.assert_frame_equal(datatuple.x, datatuple_scaled.x, check_dtype=False)
    else:
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(datatuple.x, datatuple_scaled.x, check_dtype=False)

    datatuple_post, _ = scale_continuous(dataset, datatuple_scaled, scaler2, inverse=True)

    pd.testing.assert_frame_equal(datatuple.x, datatuple_post.x, check_dtype=False)


@pytest.mark.parametrize("dataset_name", available_tabular())
@pytest.mark.parametrize("scaler_type", [StandardScaler, MinMaxScaler])
def test_scaling_separate_test(
    dataset_name: str, scaler_type: Union[Type[StandardScaler], Type[MinMaxScaler]]
) -> None:
    """Test that scaling works."""
    scaler = scaler_type()
    dataset = get_dataset_obj_by_name(dataset_name)()
    datatuple = dataset.load()

    # Speed up the tests by making some data smaller
    if dataset_name == "health":
        datatuple, _ = train_test_split(datatuple, train_percentage=0.05)

    train, test = train_test_split(datatuple)

    train_scaled, scaler2 = scale_continuous(dataset, train, scaler)
    test_scaled, _ = scale_continuous(dataset, test, scaler2, fit=False)

    if dataset_name == "crime" and str(scaler) == "MinMaxScaler()":
        # Crime dataset is minmax scaled by the data providers.
        # So can't confirm that train contains the full range
        pass
    else:
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(train.x, train_scaled.x, check_dtype=False)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(test.x, test_scaled.x, check_dtype=False)

    train_post, _ = scale_continuous(dataset, train_scaled, scaler2, inverse=True)
    test_post, _ = scale_continuous(dataset, test_scaled, scaler2, inverse=True)

    pd.testing.assert_frame_equal(train.x, train_post.x, check_dtype=False)
    pd.testing.assert_frame_equal(test.x, test_post.x, check_dtype=False)


def test_from_dummies() -> None:
    """Test that the from_dummies func produces the inverse of pd.get_dummies for an em.Dataset."""
    df = pd.DataFrame({"a": ["a", "b", "c"], "b": ["q", "w", "e"]})
    dummied = pd.get_dummies(df)
    repacked = from_dummies(dummied, {"a": ["a_a", "a_b", "a_c"], "b": ["b_q", "b_w", "b_e"]})
    pd.testing.assert_frame_equal(df, repacked)
