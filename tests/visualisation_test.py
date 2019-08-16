"""
Testing for plotting functionality
"""
import shutil
from pathlib import Path
from typing import Tuple, List

import pytest

from ethicml.algorithms.inprocess import InAlgorithm, SVM, LR, Agarwal, Kamiran, MLP, Kamishima
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess import PreAlgorithm, Upsampler
from ethicml.evaluators import evaluate_models
from ethicml.metrics import Accuracy, CV, TPR, Metric, ProbPos
from ethicml.utility import DataTuple
from ethicml.data import load_data, Adult, Dataset, Toy
from ethicml.preprocessing import train_test_split
from ethicml.visualisation import save_2d_plot, save_label_plot, save_jointplot
from ethicml.visualisation.plot import plot_mean_std_box
from tests.run_algorithm_test import get_train_test


@pytest.fixture(scope="module")
def cleanup():
    yield None
    print("remove generated directory")
    plt_dir = Path(".") / "plots"
    res_dir = Path(".") / "results"
    if plt_dir.exists():
        shutil.rmtree(plt_dir)
    if res_dir.exists():
        shutil.rmtree(res_dir)


def test_plot(cleanup):
    train, _ = get_train_test()
    save_2d_plot(train, "./plots/test.png")


def test_joint_plot(cleanup):
    train, _ = get_train_test()
    save_jointplot(train, "./plots/joint.png")


def test_label_plot(cleanup):
    data: DataTuple = load_data(Adult())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, _ = train_test

    save_label_plot(train, "./plots/labels.png")


def test_plot_evals():
    datasets: List[Dataset] = [Adult()]
    preprocess_models: List[PreAlgorithm] = [Upsampler(strategy="preferential"), Upsampler(strategy="naive"), Upsampler(strategy="uniform")]
    inprocess_models: List[InAlgorithm] = [LR(), SVM(kernel='linear'), Kamiran(), Agarwal(), MLP(), Kamishima()]
    postprocess_models: List[PostAlgorithm] = []
    metrics: List[Metric] = [Accuracy(), CV()]
    per_sens_metrics: List[Metric] = [TPR(), ProbPos()]
    results = evaluate_models(datasets, preprocess_models, inprocess_models,
                    postprocess_models, metrics, per_sens_metrics,
                    repeats=5, test_mode=False, delete_prev=True)

    plot_mean_std_box(results, Accuracy(), ProbPos())
    plot_mean_std_box(results, Accuracy(), TPR())
