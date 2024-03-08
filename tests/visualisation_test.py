"""Testing for plotting functionality."""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pytest

from ethicml import DataTuple, KernelType, Results, TrainTestPair, train_test_split
from ethicml.data import Adult, Toy, load_data
from ethicml.metrics import Accuracy, CV, NMI, ProbPos, TPR
from ethicml.models import LR, Reweighting, SVM, Upsampler, UpsampleStrategy
from ethicml.plot import (
    plot_results,
    save_2d_plot,
    save_jointplot,
    save_label_plot,
    save_multijointplot,
)
from ethicml.run import evaluate_models


@pytest.mark.slow()
@pytest.mark.usefixtures("plot_cleanup")  # fixtures are defined in `tests/conftest.py`
@pytest.mark.xdist_group("results_files")
def test_plot_tsne(toy_train_test: TrainTestPair) -> None:
    """Test plot."""
    train, _ = toy_train_test
    save_2d_plot(train, "./plots/test.png")


@pytest.mark.usefixtures("plot_cleanup")  # fixtures are defined in `tests/conftest.py`
@pytest.mark.xdist_group("results_files")
def test_plot_no_tsne(toy_train_test: TrainTestPair) -> None:
    """Test plot."""
    train, _ = toy_train_test
    train = train.replace(x=train.x[train.x.columns[:2]])
    save_2d_plot(train, "./plots/test.png")


@pytest.mark.usefixtures("plot_cleanup")
@pytest.mark.xdist_group("results_files")
def test_joint_plot(toy_train_test: TrainTestPair) -> None:
    """Test joint plot."""
    train, _ = toy_train_test
    save_jointplot(train, "./plots/joint.png")


@pytest.mark.slow()
@pytest.mark.usefixtures("plot_cleanup")
@pytest.mark.xdist_group("results_files")
def test_multijoint_plot(toy_train_test: TrainTestPair) -> None:
    """Test joint plot."""
    train, _ = toy_train_test
    save_multijointplot(train, "./plots/joint.png")


@pytest.mark.usefixtures("plot_cleanup")
@pytest.mark.xdist_group("results_files")
def test_label_plot() -> None:
    """Test label plot."""
    data: DataTuple = load_data(Adult())
    train_test: tuple[DataTuple, DataTuple] = train_test_split(data)
    train, _ = train_test

    save_label_plot(train, "./plots/labels.png")


@pytest.mark.usefixtures("plot_cleanup")
@pytest.mark.xdist_group("results_files")
def test_plot_evals() -> None:
    """Test plot evals."""
    results: Results = evaluate_models(
        datasets=[Toy()],
        preprocess_models=[Upsampler(strategy=UpsampleStrategy.preferential)],
        inprocess_models=[LR(), SVM(kernel=KernelType.linear), Reweighting()],
        metrics=[Accuracy(), CV()],
        per_sens_metrics=[TPR(), ProbPos()],
        repeats=3,
        test_mode=True,
    )
    assert results["seed"][0] == results["seed"][1] == results["seed"][2] == 0
    assert results["seed"][3] == results["seed"][4] == results["seed"][5] == 2410
    assert results["seed"][6] == results["seed"][7] == results["seed"][8] == 4820

    figs_and_plots: list[tuple[Figure, Axes]]

    # plot with metrics
    figs_and_plots = plot_results(results, Accuracy(), ProbPos())
    # num(datasets) * num(preprocess) * num(accuracy combinations) * num(prop_pos combinations)
    assert len(figs_and_plots) == 2 * 2 * 1 * 2

    # plot with column names
    figs_and_plots = plot_results(results, "Accuracy", "prob_pos_sensitive-attr_0")
    assert len(figs_and_plots) == 1 * 2 * 1 * 1

    with pytest.raises(ValueError, match='No matching columns found for Metric "NMI preds and s".'):
        plot_results(results, Accuracy(), NMI())

    with pytest.raises(ValueError, match='No column named "unknown metric".'):
        plot_results(results, "unknown metric", Accuracy())
