"""
Testing for plotting functionality
"""

from tests.run_algorithm_test import get_train_test
from visualisation.plot import save_2d_plot


def test_plot():
    train, _ = get_train_test()
    save_2d_plot(train, "plots/test.png")
