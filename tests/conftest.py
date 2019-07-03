"""Shared fixtures for testing"""
from typing import Tuple
import pytest

from ethicml.utility.data_structures import DataTuple
from ethicml.data import Toy
from ethicml.data.load import load_data
from ethicml.preprocessing.train_test_split import train_test_split


@pytest.fixture(scope="session")
def toy_train_test() -> Tuple[DataTuple, DataTuple]:
    """By making this a fixture, pytest can cache the result"""
    data: DataTuple = load_data(Toy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    return train_test
