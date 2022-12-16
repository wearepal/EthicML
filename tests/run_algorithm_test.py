"""Test that an algorithm can run against some data."""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import ethicml as em
from ethicml import ClassifierType, DataTuple, KernelType, Prediction, TestTuple, metrics, models
import ethicml.data as emda
from ethicml.models.inprocess.in_algorithm import _I, InAlgorithmDC
from ethicml.run import evaluate_models, load_results, run_in_parallel


def test_can_load_test_data(toy_train_test: em.TrainTestPair):
    """Test whether we can load test data."""
    train, test = toy_train_test
    assert train is not None
    assert test is not None


def test_run_parallel(toy_train_val: em.TrainValPair):
    """Test run parallel."""

    data0 = toy_train_val
    data1 = toy_train_val
    result = run_in_parallel(
        algos=[models.LR(), models.SVM(), models.Majority()],
        data=[em.TrainValPair(*data0), em.TrainValPair(*data1)],
        seeds=[0, 0],
        num_jobs=2,
    )
    # LR
    assert np.count_nonzero(result[0][0].hard.to_numpy() == 1) == 44
    assert np.count_nonzero(result[0][0].hard.to_numpy() == 0) == 36
    assert np.count_nonzero(result[0][1].hard.to_numpy() == 1) == 44
    assert np.count_nonzero(result[0][1].hard.to_numpy() == 0) == 36
    # SVM
    assert np.count_nonzero(result[1][0].hard.to_numpy() == 1) == 45
    assert np.count_nonzero(result[1][0].hard.to_numpy() == 0) == 35
    assert np.count_nonzero(result[1][1].hard.to_numpy() == 1) == 45
    assert np.count_nonzero(result[1][1].hard.to_numpy() == 0) == 35
    # Majority
    assert np.count_nonzero(result[2][0].hard.to_numpy() == 1) == 80
    assert np.count_nonzero(result[2][0].hard.to_numpy() == 0) == 0
    assert np.count_nonzero(result[2][1].hard.to_numpy() == 1) == 80
    assert np.count_nonzero(result[2][1].hard.to_numpy() == 0) == 0


@pytest.mark.usefixtures("results_cleanup")
@pytest.mark.xdist_group("results_files")
def test_empty_evaluate():
    """Test empty evaluate."""
    empty_result = evaluate_models([emda.Toy()], repeats=3)
    expected_result = pd.DataFrame(
        [], columns=["dataset", "scaler", "transform", "model", "split_id"]
    )
    expected_result = expected_result.set_index(
        ["dataset", "scaler", "transform", "model", "split_id"]
    )
    pd.testing.assert_frame_equal(empty_result, expected_result)


@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.usefixtures("results_cleanup")
@pytest.mark.xdist_group("results_files")
def test_run_alg_repeats_error(repeats: int):
    """Add a test to check that the right number of reults are produced with repeats."""
    dataset = emda.Adult(split=emda.Adult.Splits.RACE_BINARY)
    datasets: List[emda.Dataset] = [dataset]
    preprocess_models: List[models.PreAlgorithm] = []
    inprocess_models: List[models.InAlgorithm] = [models.LR(), models.Reweighting()]
    metrics_: List[metrics.Metric] = [metrics.Accuracy(), metrics.CV()]
    per_sens_metrics: List[metrics.Metric] = [metrics.Accuracy(), metrics.TPR()]
    results_no_scaler = evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics_,
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
@pytest.mark.xdist_group("results_files")
def test_run_repeats(repeats: int, on: Literal["data", "model", "both"]):
    """Check the repeat_on arg."""
    dataset = emda.Adult(split=emda.Adult.Splits.RACE_BINARY)
    datasets: List[emda.Dataset] = [dataset]
    preprocess_models: List[models.PreAlgorithm] = []
    inprocess_models: List[models.InAlgorithm] = [models.LR(), models.Reweighting()]
    metrics_: List[metrics.Metric] = [metrics.Accuracy(), metrics.CV()]
    per_sens_metrics: List[metrics.Metric] = [metrics.Accuracy(), metrics.TPR()]
    results_no_scaler = evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics_,
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
@pytest.mark.xdist_group("results_files")
def test_run_alg_suite_scaler():
    """Test run alg suite."""
    dataset = emda.Adult(split=emda.Adult.Splits.RACE_BINARY)
    datasets: List[emda.Dataset] = [dataset, emda.Toy()]
    preprocess_models: List[models.PreAlgorithm] = [models.Upsampler()]
    inprocess_models: List[models.InAlgorithm] = [models.LR(), models.SVM(kernel=KernelType.linear)]
    metrics_: List[metrics.Metric] = [metrics.Accuracy(), metrics.CV()]
    per_sens_metrics: List[metrics.Metric] = [metrics.Accuracy(), metrics.TPR()]
    results_no_scaler = evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics_,
        per_sens_metrics=per_sens_metrics,
        repeats=1,
        test_mode=True,
        delete_previous=True,
        topic="pytest",
    )
    results_scaler = evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics_,
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
@pytest.mark.xdist_group("results_files")
def test_run_alg_suite():
    """Test run alg suite."""
    dataset = emda.Adult(split=emda.Adult.Splits.RACE_BINARY)
    datasets: List[emda.Dataset] = [dataset, emda.Toy()]
    preprocess_models: List[models.PreAlgorithm] = [models.Upsampler()]
    inprocess_models: List[models.InAlgorithm] = [models.LR(), models.SVM(kernel=KernelType.linear)]
    metrics_: List[metrics.Metric] = [metrics.Accuracy(), metrics.CV()]
    per_sens_metrics: List[metrics.Metric] = [metrics.Accuracy(), metrics.TPR()]
    evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics_,
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

    reloaded = load_results("Adult Race-Binary", "Upsample uniform", "pytest")
    assert reloaded is not None
    read = pd.read_csv(Path(".") / "results" / "pytest_Adult Race-Binary_Upsample uniform.csv")
    read = read.set_index(["dataset", "scaler", "transform", "model", "split_id"])
    pd.testing.assert_frame_equal(reloaded, read)


@pytest.mark.usefixtures("results_cleanup")
@pytest.mark.xdist_group("results_files")
def test_run_alg_suite_wrong_metrics():
    """Test run alg suite wrong metrics."""
    datasets: List[emda.Dataset] = [emda.Toy(), emda.Adult()]
    preprocess_models: List[models.PreAlgorithm] = [models.Upsampler()]
    inprocess_models: List[models.InAlgorithm] = [models.SVM(kernel=KernelType.linear), models.LR()]
    metrics_: List[metrics.Metric] = [metrics.Accuracy(), metrics.CV()]
    per_sens_metrics: List[metrics.Metric] = [metrics.Accuracy(), metrics.TPR(), metrics.CV()]
    with pytest.raises(metrics.MetricNotApplicable):
        evaluate_models(
            datasets=datasets,
            preprocess_models=preprocess_models,
            inprocess_models=inprocess_models,
            metrics=metrics_,
            per_sens_metrics=per_sens_metrics,
            repeats=1,
            test_mode=True,
            delete_previous=False,
        )


@pytest.mark.usefixtures("results_cleanup")
@pytest.mark.xdist_group("results_files")
def test_run_alg_suite_err_handling():
    """Test run alg suite handles when an err is thrown."""

    @dataclass
    class ThrowErr(InAlgorithmDC):
        C: int = 4

        def fit(self: _I, train: DataTuple, seed: int = 888) -> _I:
            raise NotImplementedError("This won't run.")

        def predict(self, test: TestTuple) -> Prediction:
            raise NotImplementedError("This won't run.")

        def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
            raise NotImplementedError("This won't run.")

        @property
        def name(self) -> str:
            return "Problem"

    datasets: List[emda.Dataset] = [emda.Toy()]
    preprocess_models: List[models.PreAlgorithm] = []
    inprocess_models: List[models.InAlgorithm] = [models.LR(), ThrowErr()]
    metrics_: List[metrics.Metric] = [metrics.Accuracy()]
    per_sens_metrics: List[metrics.Metric] = [metrics.Accuracy()]
    results = evaluate_models(
        datasets=datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics_,
        per_sens_metrics=per_sens_metrics,
        repeats=2,
        test_mode=True,
        delete_previous=True,
    )
    assert len(results) == 4


@pytest.mark.slow
@pytest.mark.usefixtures("results_cleanup")
@pytest.mark.xdist_group("results_files")
def test_run_alg_suite_no_pipeline():
    """Run alg suite while avoiding the 'fair pipeline'."""
    datasets: List[emda.Dataset] = [emda.Toy(), emda.Adult()]
    preprocess_models: List[models.PreAlgorithm] = [models.Upsampler()]
    inprocess_models: List[models.InAlgorithm] = [
        models.Reweighting(classifier=ClassifierType.lr),
        models.LR(),
    ]
    metrics_: List[metrics.Metric] = [metrics.Accuracy(), metrics.CV()]
    per_sens_metrics: List[metrics.Metric] = [metrics.Accuracy(), metrics.TPR()]

    results = evaluate_models(
        datasets,
        preprocess_models=preprocess_models,
        inprocess_models=inprocess_models,
        metrics=metrics_,
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
