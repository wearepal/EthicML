"""Test utility functions."""
import pytest

from ethicml import data
from ethicml.utility import get_dataset_obj_by_name


@pytest.mark.parametrize("name", data.available_tabular)
def test_lookup(name):
    """Test the lookup of  a dataset by name."""
    assert name == get_dataset_obj_by_name(name).__name__
    assert name == get_dataset_obj_by_name(name)().name.lower().split()[:1][0]
