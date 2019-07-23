"""
Testing for plotting functionality
"""
from typing import Tuple

from ethicml.utility import DataTuple
from ethicml.data import load_data, Adult
from ethicml.preprocessing import train_test_split
from ethicml.visualisation import save_2d_plot, save_label_plot, save_jointplot
from tests.run_algorithm_test import get_train_test


def test_plot():
    train, _ = get_train_test()
    save_2d_plot(train, "plots/test.png")


def test_joint_plot():
    train, _ = get_train_test()
    save_jointplot(train, "plots/joint.png")


def test_label_plot():
    data: DataTuple = load_data(Adult())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, _ = train_test

    # train, _ = get_train_test()
    save_label_plot(train, "plots/labels.png")
