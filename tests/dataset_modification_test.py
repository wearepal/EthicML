"""Test modifiactions to a dataset."""
import pandas
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ethicml import data
from ethicml.data import get_dataset_obj_by_name, load_data, scale_continuous
from ethicml.preprocessing import train_test_split


@pytest.mark.parametrize("dataset_name", data.available_tabular)
@pytest.mark.parametrize("scaler", [StandardScaler, MinMaxScaler])
def test_scaling(dataset_name, scaler):
    """Test that scaling works."""
    scaler = scaler()
    dataset = get_dataset_obj_by_name(dataset_name)()
    datatuple = load_data(dataset)

    # Speed up the tests by making some data smaller
    if dataset_name == "health":
        datatuple, _ = train_test_split(datatuple, train_percentage=0.05)

    datatuple_scaled, scaler2 = scale_continuous(dataset, datatuple, scaler)

    if dataset_name == "crime" and str(scaler) == "MinMaxScaler()":
        # Crime dataset is minmax scaled by the data providers.
        pandas.testing.assert_frame_equal(datatuple.x, datatuple_scaled.x, check_dtype=False)
    else:
        with pytest.raises(AssertionError):
            pandas.testing.assert_frame_equal(datatuple.x, datatuple_scaled.x, check_dtype=False)

    datatuple_post, _ = scale_continuous(dataset, datatuple_scaled, scaler2, inverse=True)

    pandas.testing.assert_frame_equal(datatuple.x, datatuple_post.x, check_dtype=False)
