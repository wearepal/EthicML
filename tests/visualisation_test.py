"""
Testing for plotting functionality
"""
from typing import List, Tuple

import pytest
from matplotlib import pyplot as plt

from ethicml.algorithms.inprocess import LR, SVM, Kamiran
from ethicml.algorithms.preprocess import Upsampler
from ethicml.data import adult, toy, load_data
from ethicml.evaluators import evaluate_models
from ethicml.metrics import CV, NMI, TPR, Accuracy, ProbPos
from ethicml.preprocessing import train_test_split
from ethicml.utility import DataTuple, Results
from ethicml.visualisation import plot_results, save_2d_plot, save_jointplot, save_label_plot
from tests.run_algorithm_test import get_train_test


@pytest.mark.usefixtures("plot_cleanup")  # fixtures are defined in `tests/conftest.py`
def test_plot():
    """test plot"""
    train, _ = get_train_test()
    save_2d_plot(train, "./plots/test.png")


@pytest.mark.usefixtures("plot_cleanup")
def test_joint_plot():
    """test joint plot"""
    train, _ = get_train_test()
    save_jointplot(train, "./plots/joint.png")


@pytest.mark.usefixtures("plot_cleanup")
def test_label_plot():
    """test label plot"""
    data: DataTuple = load_data(adult())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, _ = train_test

    save_label_plot(train, "./plots/labels.png")


@pytest.mark.usefixtures("plot_cleanup")
def test_plot_evals():
    """test plot evals"""
    results: Results = evaluate_models(
        datasets=[adult(), toy()],
        preprocess_models=[Upsampler(strategy="preferential")],
        inprocess_models=[LR(), SVM(kernel="linear"), Kamiran()],
        metrics=[Accuracy(), CV()],
        per_sens_metrics=[TPR(), ProbPos()],
        repeats=3,
        test_mode=True,
        delete_prev=True,
    )
    assert results.data["seed"][0] == results.data["seed"][1] == results.data["seed"][2] == 0
    assert results.data["seed"][3] == results.data["seed"][4] == results.data["seed"][5] == 2410
    assert results.data["seed"][6] == results.data["seed"][7] == results.data["seed"][8] == 4820

    figs_and_plots: List[Tuple[plt.Figure, plt.Axes]]

    # plot with metrics
    figs_and_plots = plot_results(results, Accuracy(), ProbPos())
    # num(datasets) * num(preprocess) * num(accuracy combinations) * num(prop_pos combinations)
    assert len(figs_and_plots) == 2 * 2 * 1 * 2

    # plot with column names
    figs_and_plots = plot_results(results, "Accuracy", "prob_pos_s_0")
    assert len(figs_and_plots) == 1 * 2 * 1 * 1

    with pytest.raises(ValueError, match='No matching columns found for Metric "NMI preds and s".'):
        plot_results(results, Accuracy(), NMI())

    with pytest.raises(ValueError, match='No column named "unknown metric".'):
        plot_results(results, "unknown metric", Accuracy())
