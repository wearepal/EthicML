"""
Test that an algorithm can run against some data
"""
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np  # pylint: disable=unused-import  # import needed for mypy
import pandas as pd
import pytest

from ethicml.algorithms import run_blocking
from ethicml.algorithms.inprocess import LR, SVM, InAlgorithm, Kamiran, Majority
from ethicml.algorithms.postprocess import PostAlgorithm
from ethicml.algorithms.preprocess import PreAlgorithm, Upsampler
from ethicml.data import adult, Dataset, toy, load_data
from ethicml.evaluators import (
    MetricNotApplicable,
    evaluate_models,
    evaluate_models_async,
    load_results,
    run_in_parallel,
)
from ethicml.metrics import CV, TPR, Accuracy, Metric
from ethicml.preprocessing import train_test_split
from ethicml.utility import DataTuple, TrainTestPair


def get_train_test() -> Tuple[DataTuple, DataTuple]:
    """Helper function for other tests which loads the Toy dataset and splits it into train-test"""
    data: DataTuple = load_data(toy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    return train_test


def count_true(mask: "np.ndarray[np.bool_]") -> int:
    """Count the number of elements that are True"""
    return mask.nonzero()[0].shape[0]


def test_can_load_test_data():
    """test whether we can load test data"""
    train, test = get_train_test()
    assert train is not None
    assert test is not None


def test_run_parallel():
    """test run parallel"""
    data0 = get_train_test()
    data1 = get_train_test()
    result = run_blocking(
        run_in_parallel(
            [LR(), SVM(), Majority()],
            [TrainTestPair(*data0), TrainTestPair(*data1)],
            max_parallel=2,
        )
    )
    # LR
    assert count_true(result[0][0].hard.values == 1) == 211
    assert count_true(result[0][0].hard.values == -1) == 189
    assert count_true(result[0][1].hard.values == 1) == 211
    assert count_true(result[0][1].hard.values == -1) == 189
    # SVM
    assert count_true(result[1][0].hard.values == 1) == 201
    assert count_true(result[1][0].hard.values == -1) == 199
    assert count_true(result[1][1].hard.values == 1) == 201
    assert count_true(result[1][1].hard.values == -1) == 199
    # Majority
    assert count_true(result[2][0].hard.values == 1) == 0
    assert count_true(result[2][0].hard.values == -1) == 400
    assert count_true(result[2][1].hard.values == 1) == 0
    assert count_true(result[2][1].hard.values == -1) == 400


@pytest.mark.usefixtures("results_cleanup")
def test_empty_evaluate():
    """test empty evaluate"""
    empty_result = evaluate_models([toy()], repeats=3, delete_prev=True)
    expected_result = pd.DataFrame([], columns=["dataset", "transform", "model", "split_id"])
    expected_result = expected_result.set_index(["dataset", "transform", "model", "split_id"])
    pd.testing.assert_frame_equal(empty_result.data, expected_result)


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite():
    """test run alg suite"""
    dataset = adult("Race")
    dataset.sens_attrs = ["race_White"]
    datasets: List[Dataset] = [dataset, toy()]
    preprocess_models: List[PreAlgorithm] = [Upsampler()]
    inprocess_models: List[InAlgorithm] = [LR(), SVM(kernel="linear")]
    postprocess_models: List[PostAlgorithm] = []
    metrics: List[Metric] = [Accuracy(), CV()]
    per_sens_metrics: List[Metric] = [Accuracy(), TPR()]
    parallel_results = run_blocking(
        evaluate_models_async(
            datasets,
            preprocess_models,
            inprocess_models,
            postprocess_models,
            metrics,
            per_sens_metrics,
            repeats=1,
            test_mode=True,
            topic="pytest",
        )
    )
    results = evaluate_models(
        datasets,
        preprocess_models,
        inprocess_models,
        postprocess_models,
        metrics,
        per_sens_metrics,
        repeats=1,
        test_mode=True,
        delete_prev=True,
        topic="pytest",
    )
    pd.testing.assert_frame_equal(parallel_results.data, results.data, check_like=True)

    files = os.listdir(Path(".") / "results")
    file_names = [
        "pytest_Adult Race_Upsample uniform.csv",
        "pytest_Adult Race_no_transform.csv",
        "pytest_Toy_Upsample uniform.csv",
        "pytest_Toy_no_transform.csv",
    ]
    assert len(files) == 4
    assert sorted(files) == file_names

    for file in file_names:
        written_file = pd.read_csv(Path(f"./results/{file}"))
        assert (written_file["seed"][0], written_file["seed"][1]) == (0, 0)
        assert written_file.shape == (2, 15)

    reloaded = load_results("Adult Race", "Upsample uniform", "pytest")
    assert reloaded is not None
    read = pd.read_csv(Path(".") / "results" / "pytest_Adult Race_Upsample uniform.csv")
    read = read.set_index(["dataset", "transform", "model", "split_id"])
    pd.testing.assert_frame_equal(reloaded.data, read)


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite_wrong_metrics():
    """test run alg suite wrong metrics"""
    datasets: List[Dataset] = [toy(), adult()]
    preprocess_models: List[PreAlgorithm] = [Upsampler()]
    inprocess_models: List[InAlgorithm] = [SVM(kernel="linear"), LR()]
    postprocess_models: List[PostAlgorithm] = []
    metrics: List[Metric] = [Accuracy(), CV()]
    per_sens_metrics: List[Metric] = [Accuracy(), TPR(), CV()]
    with pytest.raises(MetricNotApplicable):
        evaluate_models(
            datasets,
            preprocess_models,
            inprocess_models,
            postprocess_models,
            metrics,
            per_sens_metrics,
            repeats=1,
            test_mode=True,
        )


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite_no_pipeline():
    """test run alg suite no pipeline"""
    datasets: List[Dataset] = [toy(), adult()]
    preprocess_models: List[PreAlgorithm] = [Upsampler()]
    inprocess_models: List[InAlgorithm] = [Kamiran(classifier="LR"), LR()]
    postprocess_models: List[PostAlgorithm] = []
    metrics: List[Metric] = [Accuracy(), CV()]
    per_sens_metrics: List[Metric] = [Accuracy(), TPR()]

    parallel_results = run_blocking(
        evaluate_models_async(
            datasets,
            preprocess_models,
            inprocess_models,
            postprocess_models,
            metrics,
            per_sens_metrics,
            repeats=1,
            test_mode=True,
            topic="pytest",
            fair_pipeline=False,
        )
    )
    results = evaluate_models(
        datasets,
        preprocess_models,
        inprocess_models,
        postprocess_models,
        metrics,
        per_sens_metrics,
        repeats=1,
        test_mode=True,
        topic="pytest",
        fair_pipeline=False,
        delete_prev=True,
    )
    pd.testing.assert_frame_equal(parallel_results.data, results.data, check_like=True)

    num_datasets = 2
    num_preprocess = 1
    num_fair_inprocess = 1
    num_unfair_inprocess = 1
    expected_num = num_datasets * (num_fair_inprocess + (num_preprocess + 1) * num_unfair_inprocess)
    assert len(results.data) == expected_num

    assert len(results.filter(["Kamiran & Calders LR"])) == 2  # result for Toy and for Adult
    assert len(results.filter(["Toy"], index="dataset")) == 3  # results for Kamiran, LR, Upsampler
    different_name = results.filter_and_map({"Kamiran & Calders LR": "Kamiran & Calders"})
    assert len(different_name.filter(["Kamiran & Calders LR"])) == 0
    assert len(different_name.filter(["Kamiran & Calders"])) == 2

    pd.testing.assert_frame_equal(
        results.filter(["Kamiran & Calders LR"]).data,
        results.query("model == 'Kamiran & Calders LR'").data,
    )
