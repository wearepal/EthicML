"""
Testing for plotting functionality
"""
from ethicml.visualisation.plot import save_2d_plot
from tests.run_algorithm_test import get_train_test


def test_plot():
    train, _ = get_train_test()
    save_2d_plot(train, "plots/test.png")
