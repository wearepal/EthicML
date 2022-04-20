"""Test that an algorithm can run against some data."""
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import ethicml as em
from ethicml.utility import ClassifierType, KernelType


def test_can_load_test_data(toy_train_test: em.TrainTestPair):
    """Test whether we can load test data."""
    train, test = toy_train_test
    assert train is not None
    assert test is not None


def test_run_parallel(toy_train_test: em.TrainTestPair):
    """Test run parallel."""
    from ethicml.evaluators import parallelism  # this import requires ray, so do it only on demand

    data0 = toy_train_test
    data1 = toy_train_test
    result = parallelism.run_in_parallel(
        [em.LR(), em.SVM(), em.Majority()],
        [em.TrainTestPair(*data0), em.TrainTestPair(*data1)],
        num_jobs=2,
    )
    # LR
    assert np.count_nonzero(result[0][0].hard.values == 1) == 44
    assert np.count_nonzero(result[0][0].hard.values == 0) == 36
    assert np.count_nonzero(result[0][1].hard.values == 1) == 44
    assert np.count_nonzero(result[0][1].hard.values == 0) == 36
    # SVM
    assert np.count_nonzero(result[1][0].hard.values == 1) == 45
    assert np.count_nonzero(result[1][0].hard.values == 0) == 35
    assert np.count_nonzero(result[1][1].hard.values == 1) == 45
    assert np.count_nonzero(result[1][1].hard.values == 0) == 35
    # Majority
    assert np.count_nonzero(result[2][0].hard.values == 1) == 80
    assert np.count_nonzero(result[2][0].hard.values == 0) == 0
    assert np.count_nonzero(result[2][1].hard.values == 1) == 80
    assert np.count_nonzero(result[2][1].hard.values == 0) == 0


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
    inprocess_models: List[em.InAlgorithm] = [em.LR(), em.SVM(kernel=KernelType.linear)]
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR()]
    results_no_scaler = em.evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics,
        per_sens_metrics=per_sens_metrics,
        repeats=1,
        test_mode=True,
        delete_prev=True,
        topic="pytest",
    )
    results_scaler = em.evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics,
        per_sens_metrics=per_sens_metrics,
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
    inprocess_models: List[em.InAlgorithm] = [em.LR(), em.SVM(kernel=KernelType.linear)]
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR()]
    results = em.evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics,
        per_sens_metrics=per_sens_metrics,
        repeats=1,
        test_mode=True,
        delete_prev=True,
        topic="pytest",
    )

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
        assert written_file.shape == (2, 18)

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
    inprocess_models: List[em.InAlgorithm] = [em.SVM(kernel=KernelType.linear), em.LR()]
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR(), em.CV()]
    with pytest.raises(em.MetricNotApplicable):
        em.evaluate_models(
            datasets=datasets,
            preprocess_models=preprocess_models,
            inprocess_models=inprocess_models,
            metrics=metrics,
            per_sens_metrics=per_sens_metrics,
            repeats=1,
            test_mode=True,
        )


@pytest.mark.slow
@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite_no_pipeline():
    """Run alg suite while avoiding the 'fair pipeline'."""
    datasets: List[em.Dataset] = [em.toy(), em.adult()]
    preprocess_models: List[em.PreAlgorithm] = [em.Upsampler()]
    inprocess_models: List[em.InAlgorithm] = [em.Kamiran(classifier=ClassifierType.lr), em.LR()]
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR()]

    results = em.evaluate_models(
        datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics,
        per_sens_metrics=per_sens_metrics,
        repeats=1,
        test_mode=True,
        topic="pytest",
        fair_pipeline=False,
        delete_prev=True,
    )

    num_datasets = 2
    num_preprocess = 1
    num_fair_inprocess = 1
    num_unfair_inprocess = 1
    expected_num = num_datasets * (num_fair_inprocess + (num_preprocess + 1) * num_unfair_inprocess)
    assert len(results) == expected_num

    kc_name = "Kamiran & Calders LR C=1.0"

    assert len(em.filter_results(results, [kc_name])) == 2  # result for Toy and Adult
    assert (
        len(em.filter_results(results, ["Toy"], index="dataset")) == 3
    )  # Kamiran, LR and Upsampler
    different_name = em.filter_and_map_results(results, {kc_name: "Kamiran & Calders"})
    assert len(em.filter_results(different_name, [kc_name])) == 0
    assert len(em.filter_results(different_name, ["Kamiran & Calders"])) == 2

    pd.testing.assert_frame_equal(
        em.filter_results(results, [kc_name]),
        results.query(f"model == '{kc_name}'"),
    )
