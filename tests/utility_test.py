"""Test utility functions."""
import pytest

from ethicml import available_tabular, get_dataset_obj_by_name


@pytest.mark.parametrize("name", available_tabular())
def test_lookup(name: str):
    """Test the lookup of  a dataset by name."""
    assert name == get_dataset_obj_by_name(name).__name__.lower()
    assert name.replace("_", "") == get_dataset_obj_by_name(name)().name.lower().split()[:1][0]
