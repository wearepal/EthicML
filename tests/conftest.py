"""
This file is automatically imported by pytest (no need to import it) and defines shared fixtures
"""
from pathlib import Path
import shutil

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


@pytest.fixture(scope="module")
def plot_cleanup():
    """
    Clean up after the tests by removing the `plots` and `results` directories
    """
    yield None
    print("remove generated directory")
    plt_dir = Path(".") / "plots"
    res_dir = Path(".") / "results"
    if plt_dir.exists():
        shutil.rmtree(plt_dir)
    if res_dir.exists():
        shutil.rmtree(res_dir)


@pytest.fixture(scope="function")
def results_cleanup():
    """
    Clean up after the tests by removing the `results` directory
    """
    yield None
    print("remove generated directory")
    res_dir = Path(".") / "results"
    if res_dir.exists():
        shutil.rmtree(res_dir)
