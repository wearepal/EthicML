"""Test that an algorithm can run against some data."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing_extensions import Literal

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import ethicml as em
from ethicml import DataTuple, Prediction, TestTuple
from ethicml.algorithms.inprocess.in_algorithm import _I, InAlgorithmDC
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
        algos=[em.LR(), em.SVM(), em.Majority()],
        data=[em.TrainTestPair(*data0), em.TrainTestPair(*data1)],
        seeds=[0, 0],
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
    empty_result = em.evaluate_models([em.Toy()], repeats=3)
    expected_result = pd.DataFrame(
        [], columns=["dataset", "scaler", "transform", "model", "split_id"]
    )
    expected_result = expected_result.set_index(
        ["dataset", "scaler", "transform", "model", "split_id"]
    )
    pd.testing.assert_frame_equal(empty_result, expected_result)


@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_repeats_error(repeats: int):
    """Add a test to check that the right number of reults are produced with repeats."""
    dataset = em.Adult(split=em.Adult.Splits.RACE_BINARY)
    datasets: List[em.Dataset] = [dataset]
    preprocess_models: List[em.PreAlgorithm] = []
    inprocess_models: List[em.InAlgorithm] = [em.LR(), em.Kamiran()]
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR()]
    results_no_scaler = em.evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics,
        per_sens_metrics=per_sens_metrics,
        repeats=repeats,
        test_mode=True,
        delete_previous=True,
        topic="pytest",
    )
    assert len(results_no_scaler) == repeats * len(inprocess_models)


@pytest.mark.parametrize("on", ["data", "model", "both"])
@pytest.mark.parametrize("repeats", [2, 3, 5])
@pytest.mark.usefixtures("results_cleanup")
def test_run_repeats(repeats: int, on: Literal["data", "model", "both"]):
    """Check the repeat_on arg."""
    dataset = em.Adult(split=em.Adult.Splits.RACE_BINARY)
    datasets: List[em.Dataset] = [dataset]
    preprocess_models: List[em.PreAlgorithm] = []
    inprocess_models: List[em.InAlgorithm] = [em.LR(), em.Kamiran()]
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR()]
    results_no_scaler = em.evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics,
        per_sens_metrics=per_sens_metrics,
        repeats=repeats,
        test_mode=True,
        delete_previous=True,
        repeat_on=on,
        topic="pytest",
    )
    assert len(results_no_scaler) == repeats * len(inprocess_models)
    if on == "data":
        assert (results_no_scaler["model_seed"] == 0).all()
        assert results_no_scaler["seed"].sum() > 0
    elif on == "model":
        assert (results_no_scaler["seed"] == 0).all()
        assert results_no_scaler["model_seed"].sum() > 0
    else:
        assert (results_no_scaler["seed"] == results_no_scaler["model_seed"] * 2410).all()


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite_scaler():
    """Test run alg suite."""
    dataset = em.Adult(split=em.Adult.Splits.RACE_BINARY)
    datasets: List[em.Dataset] = [dataset, em.Toy()]
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
        delete_previous=True,
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
        delete_previous=True,
        topic="pytest",
    )
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(results_scaler, results_no_scaler, check_like=True)


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite():
    """Test run alg suite."""
    dataset = em.Adult(split=em.Adult.Splits.RACE_BINARY)
    datasets: List[em.Dataset] = [dataset, em.Toy()]
    preprocess_models: List[em.PreAlgorithm] = [em.Upsampler()]
    inprocess_models: List[em.InAlgorithm] = [em.LR(), em.SVM(kernel=KernelType.linear)]
    metrics: List[em.Metric] = [em.Accuracy(), em.CV()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy(), em.TPR()]
    em.evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics,
        per_sens_metrics=per_sens_metrics,
        repeats=1,
        test_mode=True,
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
        assert written_file.shape == (2, 19)

    reloaded = em.load_results("Adult Race-Binary", "Upsample uniform", "pytest")
    assert reloaded is not None
    read = pd.read_csv(Path(".") / "results" / "pytest_Adult Race-Binary_Upsample uniform.csv")
    read = read.set_index(["dataset", "scaler", "transform", "model", "split_id"])
    pd.testing.assert_frame_equal(reloaded, read)


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite_wrong_metrics():
    """Test run alg suite wrong metrics."""
    datasets: List[em.Dataset] = [em.Toy(), em.Adult()]
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
            delete_previous=False,
        )


@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite_err_handling():
    """Test run alg suite handles when an err is thrown."""

    @dataclass
    class ThrowErr(InAlgorithmDC):
        C: int = 4

        def fit(self: _I, train: DataTuple, seed: int = 888) -> _I:
            pass

        def predict(self, test: TestTuple) -> Prediction:
            pass

        def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
            raise NotImplementedError("This won't run.")

        def get_name(self) -> str:
            return "Problem"

    datasets: List[em.Dataset] = [em.Toy()]
    preprocess_models: List[em.PreAlgorithm] = []
    inprocess_models: List[em.InAlgorithm] = [em.LR(), ThrowErr()]
    metrics: List[em.Metric] = [em.Accuracy()]
    per_sens_metrics: List[em.Metric] = [em.Accuracy()]
    results = em.evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics,
        per_sens_metrics=per_sens_metrics,
        repeats=2,
        test_mode=True,
        delete_previous=True,
    )
    assert len(results) == 4


@pytest.mark.slow
@pytest.mark.usefixtures("results_cleanup")
def test_run_alg_suite_no_pipeline():
    """Run alg suite while avoiding the 'fair pipeline'."""
    datasets: List[em.Dataset] = [em.Toy(), em.Adult()]
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
    )

    num_datasets = 2
    num_preprocess = 1
    num_fair_inprocess = 1
    num_unfair_inprocess = 1
    expected_num = num_datasets * (num_fair_inprocess + (num_preprocess + 1) * num_unfair_inprocess)
    assert len(results) == expected_num

    kc_name = "Kamiran & Calders lr C=1.0"

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
