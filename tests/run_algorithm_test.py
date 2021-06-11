"""Test that an algorithm can run against some data."""
import os
from pathlib import Path
from typing import List

import numpy as np  # pylint: disable=unused-import  # import needed for mypy
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import ethicml as em


def count_true(mask: np.ndarray) -> int:
    """Count the number of elements that are True."""
    return mask.nonzero()[0].shape[0]


def test_can_load_test_data(toy_train_test: em.TrainTestPair):
    """Test whether we can load test data."""
    train, test = toy_train_test
    assert train is not None
    assert test is not None


def test_run_parallel(toy_train_test: em.TrainTestPair):
    """Test run parallel."""
    data0 = toy_train_test
    data1 = toy_train_test
    result = em.run_blocking(
        em.run_in_parallel(
            [em.LR(), em.SVM(), em.Majority()],
            [em.TrainTestPair(*data0), em.TrainTestPair(*data1)],
            max_parallel=2,
        )
    )
    # LR
    assert count_true(result[0][0].hard.values == 1) == 44
    assert count_true(result[0][0].hard.values == 0) == 36
    assert count_true(result[0][1].hard.values == 1) == 44
    assert count_true(result[0][1].hard.values == 0) == 36
    # SVM
    assert count_true(result[1][0].hard.values == 1) == 45
    assert count_true(result[1][0].hard.values == 0) == 35
    assert count_true(result[1][1].hard.values == 1) == 45
    assert count_true(result[1][1].hard.values == 0) == 35
    # Majority
    assert count_true(result[2][0].hard.values == 1) == 80
    assert count_true(result[2][0].hard.values == 0) == 0
    assert count_true(result[2][1].hard.values == 1) == 80
    assert count_true(result[2][1].hard.values == 0) == 0


@pytest.mark.usefixtures("results_cleanup")
def test_empty_evaluate():
    """Test empty evaluate."""
    empty_result = em.evaluate_models([em.toy()], repeats=3, delete_prev=True)
    expected_result = pd.DataFrame(
        [], columns=["dataset", "scaler", "transform", "model", "split_id"]
    )
    expected_result = expected_result.set_index(
        ["dataset", "scaler", "transform", "model", "split_id"]
    )
    pd.testing.assert_frame_equal(empty_result, expected_result)


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite_scaler():
    """Test run alg suite."""
    dataset = em.adult(split="Race-Binary")
    datasets: List[em.Dataset] = [dataset, em.toy()]
    preprocess_models: List[em.PreAlgorithm] = [em.Upsampler()]
    inprocess_models: List[em.InAlgorithm] = [em.LR(), em.SVM(kernel="linear")]
    postprocess_models: List[em.PostAlgorithm] = []
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR()]
    results_no_scaler = em.evaluate_models(
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
    results_scaler = em.evaluate_models(
        datasets,
        preprocess_models,
        inprocess_models,
        postprocess_models,
        metrics,
        per_sens_metrics,
        scaler=StandardScaler(),
        repeats=1,
        test_mode=True,
        delete_prev=True,
        topic="pytest",
    )
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(results_scaler, results_no_scaler, check_like=True)


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite():
    """Test run alg suite."""
    dataset = em.adult(split="Race-Binary")
    datasets: List[em.Dataset] = [dataset, em.toy()]
    preprocess_models: List[em.PreAlgorithm] = [em.Upsampler()]
    inprocess_models: List[em.InAlgorithm] = [em.LR(), em.SVM(kernel="linear")]
    postprocess_models: List[em.PostAlgorithm] = []
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR()]
    parallel_results = em.run_blocking(
        em.evaluate_models_async(
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
    results = em.evaluate_models(
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
    pd.testing.assert_frame_equal(parallel_results, results, check_like=True)

    files = os.listdir(Path(".") / "results")
    file_names = [
        "pytest_Adult Race-Binary_Upsample uniform.csv",
        "pytest_Adult Race-Binary_no_transform.csv",
        "pytest_Toy_Upsample uniform.csv",
        "pytest_Toy_no_transform.csv",
    ]
    assert len(files) == 4
    assert sorted(files) == file_names

    for file in file_names:
        written_file = pd.read_csv(Path(f"./results/{file}"))
        assert (written_file["seed"][0], written_file["seed"][1]) == (0, 0)
        assert written_file.shape == (2, 16)

    reloaded = em.load_results("Adult Race-Binary", "Upsample uniform", "pytest")
    assert reloaded is not None
    read = pd.read_csv(Path(".") / "results" / "pytest_Adult Race-Binary_Upsample uniform.csv")
    read = read.set_index(["dataset", "scaler", "transform", "model", "split_id"])
    pd.testing.assert_frame_equal(reloaded, read)


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite_wrong_metrics():
    """Test run alg suite wrong metrics."""
    datasets: List[em.Dataset] = [em.toy(), em.adult()]
    preprocess_models: List[em.PreAlgorithm] = [em.Upsampler()]
    inprocess_models: List[em.InAlgorithm] = [em.SVM(kernel="linear"), em.LR()]
    postprocess_models: List[em.PostAlgorithm] = []
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR(), em.CV()]
    with pytest.raises(em.MetricNotApplicable):
        em.evaluate_models(
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
    """Test run alg suite no pipeline."""
    datasets: List[em.Dataset] = [em.toy(), em.adult()]
    preprocess_models: List[em.PreAlgorithm] = [em.Upsampler()]
    inprocess_models: List[em.InAlgorithm] = [em.Kamiran(classifier="LR"), em.LR()]
    postprocess_models: List[em.PostAlgorithm] = []
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR()]

    parallel_results = em.run_blocking(
        em.evaluate_models_async(
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
    results = em.evaluate_models(
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
    pd.testing.assert_frame_equal(parallel_results, results, check_like=True)

    num_datasets = 2
    num_preprocess = 1
    num_fair_inprocess = 1
    num_unfair_inprocess = 1
    expected_num = num_datasets * (num_fair_inprocess + (num_preprocess + 1) * num_unfair_inprocess)
    assert len(results) == expected_num

    assert (
        len(em.filter_results(results, ["Kamiran & Calders LR"])) == 2
    )  # result for Toy and Adult
    assert (
        len(em.filter_results(results, ["Toy"], index="dataset")) == 3
    )  # Kamiran, LR and Upsampler
    different_name = em.filter_and_map_results(
        results, {"Kamiran & Calders LR": "Kamiran & Calders"}
    )
    assert len(em.filter_results(different_name, ["Kamiran & Calders LR"])) == 0
    assert len(em.filter_results(different_name, ["Kamiran & Calders"])) == 2

    pd.testing.assert_frame_equal(
        em.filter_results(results, ["Kamiran & Calders LR"]),
        results.query("model == 'Kamiran & Calders LR'"),
    )
