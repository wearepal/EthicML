"""
Test that an algorithm can run against some data
"""
import os
import shutil
from pathlib import Path
from typing import Tuple, List
import numpy as np  # import needed for mypy
import pandas as pd
import pytest

from ethicml.algorithms.inprocess import LR, SVM, Majority, InAlgorithm
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess import PreAlgorithm, Upsampler
from ethicml.metrics import Accuracy, CV, TPR, Metric
from ethicml.utility import DataTuple, TrainTestPair
from ethicml.data import load_data, Toy, Adult, Dataset
from ethicml.evaluators import (
    run_in_parallel,
    evaluate_models,
    MetricNotApplicable,
    load_results,
    evaluate_models_parallel,
)
from ethicml.preprocessing import train_test_split


def get_train_test() -> Tuple[DataTuple, DataTuple]:
    data: DataTuple = load_data(Toy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    return train_test


def count_true(mask: 'np.ndarray[bool]') -> int:
    """Count the number of elements that are True"""
    return mask.nonzero()[0].shape[0]


def test_can_load_test_data():
    train, test = get_train_test()
    assert train is not None
    assert test is not None


def test_run_parallel():
    data0 = get_train_test()
    data1 = get_train_test()
    result = run_in_parallel(
        [LR(), SVM(), Majority()],
        [TrainTestPair(*data0), TrainTestPair(*data1)],
        max_parallel=2
    )
    # LR
    assert count_true(result[0][0].values == 1) == 211
    assert count_true(result[0][0].values == -1) == 189
    assert count_true(result[0][1].values == 1) == 211
    assert count_true(result[0][1].values == -1) == 189
    # SVM
    assert count_true(result[1][0].values == 1) == 201
    assert count_true(result[1][0].values == -1) == 199
    assert count_true(result[1][1].values == 1) == 201
    assert count_true(result[1][1].values == -1) == 199
    # Majority
    assert count_true(result[2][0].values == 1) == 0
    assert count_true(result[2][0].values == -1) == 400
    assert count_true(result[2][1].values == 1) == 0
    assert count_true(result[2][1].values == -1) == 400


@pytest.fixture(scope="function")
def cleanup():
    yield None
    print("remove generated directory")
    res_dir = Path(".") / "results"
    if res_dir.exists():
        shutil.rmtree(res_dir)


def test_empty_evaluate(cleanup):
    empty_result = evaluate_models([Toy()], repeats=3)
    expected_result = pd.DataFrame([], columns=['dataset', 'transform', 'model', 'repeat'])
    expected_result = expected_result.set_index(["dataset", "transform", "model", "repeat"])
    pd.testing.assert_frame_equal(empty_result.data, expected_result)


def test_run_alg_suite(cleanup):
    dataset = Adult("Race")
    dataset.sens_attrs = ["race_White"]
    datasets: List[Dataset] = [dataset, Toy()]
    preprocess_models: List[PreAlgorithm] = [Upsampler()]
    inprocess_models: List[InAlgorithm] = [LR(), SVM(kernel='linear')]
    postprocess_models: List[PostAlgorithm] = []
    metrics: List[Metric] = [Accuracy(), CV()]
    per_sens_metrics: List[Metric] = [Accuracy(), TPR()]
    parallel_results = evaluate_models_parallel(datasets, inprocess_models, metrics,
                                                per_sens_metrics, repeats=1, test_mode=True,
                                                topic="pytest")
    results = evaluate_models(datasets, preprocess_models, inprocess_models,
                              postprocess_models, metrics, per_sens_metrics,
                              repeats=1, test_mode=True, delete_prev=True, topic="pytest")
    pd.testing.assert_frame_equal(
        parallel_results.data, results.data.query("transform == \"no_transform\"")
    )

    files = os.listdir(Path(".") / "results")
    file_names = [
        'pytest_Adult Race_Upsample uniform.csv',
        'pytest_Adult Race_no_transform.csv',
        'pytest_Toy_Upsample uniform.csv',
        'pytest_Toy_no_transform.csv',
    ]
    assert len(files) == 4
    assert sorted(files) == file_names

    for file in file_names:
        written_file = pd.read_csv(Path(f"./results/{file}"))
        assert written_file.shape == (2, 14)

    reloaded = load_results("Adult Race", "Upsample uniform", "pytest")
    assert reloaded is not None
    read = pd.read_csv(Path(".") / "results" / 'pytest_Adult Race_Upsample uniform.csv')
    read = read.set_index(["dataset", "transform", "model", "repeat"])
    pd.testing.assert_frame_equal(reloaded.data, read)


def test_run_alg_suite_wrong_metrics(cleanup):
    datasets: List[Dataset] = [Toy(), Adult()]
    preprocess_models: List[PreAlgorithm] = [Upsampler()]
    inprocess_models: List[InAlgorithm] = [SVM(kernel='linear'), LR()]
    postprocess_models: List[PostAlgorithm] = []
    metrics: List[Metric] = [Accuracy(), CV()]
    per_sens_metrics: List[Metric] = [Accuracy(), TPR(), CV()]
    with pytest.raises(MetricNotApplicable):
        evaluate_models(datasets, preprocess_models, inprocess_models,
                        postprocess_models, metrics, per_sens_metrics, repeats=1, test_mode=True)
