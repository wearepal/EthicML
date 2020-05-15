"""
Testing for plotting functionality
"""
from typing import List, Tuple

import pytest
from matplotlib import pyplot as plt

from ethicml.algorithms.inprocess import LR, SVM, Kamiran
from ethicml.algorithms.preprocess import Upsampler
from ethicml.data import adult, load_data, toy
from ethicml.evaluators import evaluate_models
from ethicml.metrics import CV, NMI, TPR, Accuracy, ProbPos
from ethicml.preprocessing import train_test_split
from ethicml.utility import DataTuple, Results, TrainTestPair
from ethicml.visualisation import plot_results, save_2d_plot, save_jointplot, save_label_plot


@pytest.mark.usefixtures("plot_cleanup")  # fixtures are defined in `tests/conftest.py`
def test_plot(toy_train_test: TrainTestPair):
    """test plot"""
    train, _ = toy_train_test
    save_2d_plot(train, "./plots/test.png")


@pytest.mark.usefixtures("plot_cleanup")
def test_joint_plot(toy_train_test: TrainTestPair):
    """test joint plot"""
    train, _ = toy_train_test
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
    assert results["seed"][0] == results["seed"][1] == results["seed"][2] == 0
    assert results["seed"][3] == results["seed"][4] == results["seed"][5] == 2410
    assert results["seed"][6] == results["seed"][7] == results["seed"][8] == 4820

    figs_and_plots: List[Tuple[plt.Figure, plt.Axes]]

    # plot with metrics
    figs_and_plots = plot_results(results, Accuracy(), ProbPos())
    # num(datasets) * num(preprocess) * num(accuracy combinations) * num(prop_pos combinations)
    assert len(figs_and_plots) == 2 * 2 * 1 * 2 + 4  # TODO: this +4 should be FIXED,
    # it matches the column name containing a hyphen as a DIFF metric.

    # plot with column names
    figs_and_plots = plot_results(results, "Accuracy", "prob_pos_sensitive-attr_0")
    assert len(figs_and_plots) == 1 * 2 * 1 * 1

    with pytest.raises(ValueError, match='No matching columns found for Metric "NMI preds and s".'):
        plot_results(results, Accuracy(), NMI())

    with pytest.raises(ValueError, match='No column named "unknown metric".'):
        plot_results(results, "unknown metric", Accuracy())
