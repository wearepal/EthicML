"""Shared fixtures for testing"""
from typing import Tuple
import pytest

from ethicml.utility import DataTuple
from ethicml.data import load_data, Toy
from ethicml.preprocessing import train_test_split


@pytest.fixture(scope="session")
def toy_train_test() -> Tuple[DataTuple, DataTuple]:
    """By making this a fixture, pytest can cache the result"""
    data: DataTuple = load_data(Toy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    return train_test
