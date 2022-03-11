"""Test modifiactions to a dataset."""
import pandas
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ethicml import (
    Dataset,
    available_tabular,
    get_dataset_obj_by_name,
    scale_continuous,
    train_test_split,
)


@pytest.mark.parametrize("dataset_name", available_tabular())
@pytest.mark.parametrize("scaler", [StandardScaler, MinMaxScaler])
def test_scaling(dataset_name, scaler):
    """Test that scaling works."""
    scaler = scaler()
    dataset = get_dataset_obj_by_name(dataset_name)()
    datatuple = dataset.load()

    # Speed up the tests by making some data smaller
    if dataset_name == "health":
        datatuple, _ = train_test_split(datatuple, train_percentage=0.05)

    datatuple_scaled, scaler2 = scale_continuous(dataset, datatuple, scaler)

    if dataset_name == "crime" and str(scaler) == "MinMaxScaler()":
        # Crime dataset is minmax scaled by the data providers.
        pandas.testing.assert_frame_equal(datatuple.x, datatuple_scaled.x, check_dtype=False)  # type: ignore[call-arg]
    else:
        with pytest.raises(AssertionError):
            pandas.testing.assert_frame_equal(datatuple.x, datatuple_scaled.x, check_dtype=False)  # type: ignore[call-arg]

    datatuple_post, _ = scale_continuous(dataset, datatuple_scaled, scaler2, inverse=True)

    pandas.testing.assert_frame_equal(datatuple.x, datatuple_post.x, check_dtype=False)  # type: ignore[call-arg]


@pytest.mark.parametrize("dataset_name", available_tabular())
@pytest.mark.parametrize("scaler", [StandardScaler, MinMaxScaler])
def test_scaling_separate_test(dataset_name, scaler):
    """Test that scaling works."""
    scaler = scaler()
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
            pandas.testing.assert_frame_equal(train.x, train_scaled.x, check_dtype=False)  # type: ignore[call-arg]
        with pytest.raises(AssertionError):
            pandas.testing.assert_frame_equal(test.x, test_scaled.x, check_dtype=False)  # type: ignore[call-arg]

    train_post, _ = scale_continuous(dataset, train_scaled, scaler2, inverse=True)
    test_post, _ = scale_continuous(dataset, test_scaled, scaler2, inverse=True)

    pandas.testing.assert_frame_equal(train.x, train_post.x, check_dtype=False)  # type: ignore[call-arg]
    pandas.testing.assert_frame_equal(test.x, test_post.x, check_dtype=False)  # type: ignore[call-arg]


def test_from_dummies():
    """Test that the _from_dummies method produces the inverse of pd.get_dummies for an em.Datase."""
    df = pd.DataFrame({"a": ["a", "b", "c"], "b": ["q", "w", "e"]})
    dummied = pd.get_dummies(df)
    repacked = Dataset._from_dummies(
        dummied, {"a": ["a_a", "a_b", "a_c"], "b": ["b_q", "b_w", "b_e"]}
    )
    pandas.testing.assert_frame_equal(df, repacked)
