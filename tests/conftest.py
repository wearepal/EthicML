"""Shared fixtures for testing"""
import pytest

from ethicml.utility import DataTuple, TrainTestPair
from ethicml.data import load_data, Toy
from ethicml.preprocessing import train_test_split


@pytest.fixture(scope="session")
def toy_train_test() -> TrainTestPair:
    """By making this a fixture, pytest can cache the result"""
    data: DataTuple = load_data(Toy())
    train: DataTuple
    test: DataTuple
    train, test = train_test_split(data)
    return TrainTestPair(train, test.remove_y())
