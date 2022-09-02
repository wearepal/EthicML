"""Test that we can get some metrics on predictions."""

from typing import Tuple

import pytest
from pytest import approx

from ethicml.run import run_metrics
from ethicml import (
    BalancedTestSplit,
    DataTuple,
    EvalTuple,
    Prediction,
    TrainValPair,
    train_test_split,
)
from ethicml.data import Adult, NonBinaryToy, load_data
from ethicml.metrics import (
    Accuracy,
    BCR,
    CV,
    LabelOutOfBounds,
    MetricNotApplicable,
    NMI,
    NPV,
    PerSens,
    PPV,
    ProbPos,
    RenyiCorrelation,
    TNR,
    TPR,
    Yanovich,
    diff_per_sens,
    metric_per_sens,
    ratio_per_sens,
)
from ethicml.models import InAlgorithm, LR, LRCV, SVM


def test_tpr_diff(toy_train_val: TrainValPair):
    """Test tpr diff."""
    train, test = toy_train_val
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run(train, test)
    tprs = metric_per_sens(predictions, test, TPR())
    assert TPR().name == "TPR"
    assert tprs == {
        "sensitive-attr_0": approx(0.923, abs=0.001),
        "sensitive-attr_1": approx(1.0, abs=0.001),
    }
    tpr_diff = diff_per_sens(tprs)
    print(tpr_diff)
    assert tpr_diff["sensitive-attr_0-sensitive-attr_1"] == approx(0.077, abs=0.001)


def test_ppv_diff(toy_train_val: TrainValPair):
    """Test ppv diff."""
    train, test = toy_train_val
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run(train, test)
    results = metric_per_sens(predictions, test, PPV())
    assert PPV().name == "PPV"
    assert results == {
        "sensitive-attr_0": approx(0.857, abs=0.001),
        "sensitive-attr_1": approx(0.903, abs=0.001),
    }
    diff = diff_per_sens(results)
    assert diff["sensitive-attr_0-sensitive-attr_1"] == approx(0.05, abs=0.1)


def test_npv_diff(toy_train_val: TrainValPair):
    """Test npv diff."""
    train, test = toy_train_val
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run(train, test)
    results = metric_per_sens(predictions, test, NPV())
    assert NPV().name == "NPV"
    assert results == {
        "sensitive-attr_0": approx(0.958, abs=0.001),
        "sensitive-attr_1": approx(1.0, abs=0.001),
    }
    diff = diff_per_sens(results)
    assert diff["sensitive-attr_0-sensitive-attr_1"] == approx(0.042, abs=0.001)


def test_bcr_diff(toy_train_val: TrainValPair):
    """Test bcr diff."""
    train, test = toy_train_val
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run(train, test)
    results = metric_per_sens(predictions, test, BCR())
    assert BCR().name == "BCR"
    assert results == {
        "sensitive-attr_0": approx(0.921, abs=0.001),
        "sensitive-attr_1": approx(0.892, abs=0.001),
    }
    diff = diff_per_sens(results)
    assert diff["sensitive-attr_0-sensitive-attr_1"] == approx(0.029, abs=0.001)


def test_use_appropriate_metric(toy_train_val: TrainValPair):
    """Test use appropriate metric."""
    train, test = toy_train_val
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run(train, test)
    with pytest.raises(MetricNotApplicable):
        metric_per_sens(predictions, test, CV())


def test_run_metrics(toy_train_val: TrainValPair):
    """Test run metrics."""
    train, test = toy_train_val
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run(train, test)
    results = run_metrics(predictions, test, [CV()], [TPR()])
    assert len(results) == 5
    assert results["TPR_sensitive-attr_0"] == approx(0.923, abs=0.001)
    assert results["TPR_sensitive-attr_1"] == approx(1.0, abs=0.001)
    assert results["TPR_sensitive-attr_0-sensitive-attr_1"] == approx(0.077, abs=0.001)
    assert results["TPR_sensitive-attr_0÷sensitive-attr_1"] == approx(0.923, abs=0.001)
    assert results["CV"] == approx(0.630, abs=0.001)
    results = run_metrics(predictions, test, [], [TPR()], {PerSens.MIN, PerSens.DIFFS})
    assert len(results) == 4
    assert results["TPR_sensitive-attr_0"] == approx(0.923, abs=0.001)
    assert results["TPR_sensitive-attr_1"] == approx(1.0, abs=0.001)
    assert results["TPR_sensitive-attr_0-sensitive-attr_1"] == approx(0.077, abs=0.001)
    assert results["TPR_min(sensitive-attr_0,sensitive-attr_1)"] == approx(0.923, abs=0.001)
    results = run_metrics(
        predictions, test, [], [TPR()], {PerSens.MAX, PerSens.RATIOS}, use_sens_name=False
    )
    assert len(results) == 4
    assert results["TPR_S_0"] == approx(0.923, abs=0.001)
    assert results["TPR_S_1"] == approx(1.0, abs=0.001)
    assert results["TPR_S_0÷S_1"] == approx(0.923, abs=0.001)
    assert results["TPR_max(S_0,S_1)"] == approx(1.0, abs=0.001)
    results = run_metrics(predictions, test, [], [TPR()], aggregation=set(), use_sens_name=False)
    assert len(results) == 2
    assert results["TPR_S_0"] == approx(0.923, abs=0.001)
    assert results["TPR_S_1"] == approx(1.0, abs=0.001)


def test_get_info(toy_train_val: TrainValPair):
    """Test get info."""
    train, test = toy_train_val
    model: LRCV = LRCV()
    predictions: Prediction = model.run(train, test)
    assert predictions.info["C"] == approx(166.810, abs=0.001)


def test_tpr_diff_non_binary_race():
    """Test tpr diff non binary race."""
    data: DataTuple = load_data(Adult(split=Adult.Splits.RACE))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run_test(train, test)
    tprs = metric_per_sens(predictions, test, TPR())
    assert TPR().name == "TPR"
    test_dict = {
        "race_0": approx(0.37, abs=0.01),
        "race_1": approx(0.12, abs=0.01),
        "race_2": approx(0.14, abs=0.01),
        "race_3": approx(0.12, abs=0.01),
        "race_4": approx(0.16, abs=0.01),
    }

    for key, val in tprs.items():
        assert val == test_dict[key]

    tpr_diff = diff_per_sens(tprs)
    test_dict = {
        "race_0-race_1": approx(0.25, abs=0.01),
        "race_0-race_2": approx(0.23, abs=0.01),
        "race_0-race_3": approx(0.25, abs=0.01),
        "race_0-race_4": approx(0.20, abs=0.01),
        "race_1-race_2": approx(0.01, abs=0.01),
        "race_1-race_3": approx(0.00, abs=0.01),
        "race_1-race_4": approx(0.04, abs=0.01),
        "race_2-race_3": approx(0.01, abs=0.01),
        "race_2-race_4": approx(0.04, abs=0.01),
        "race_3-race_4": approx(0.04, abs=0.01),
    }

    for key, val in tpr_diff.items():
        assert val == test_dict[key]


def test_tpr_ratio_non_binary_race():
    """Test tpr ratio non binary race."""
    data: DataTuple = load_data(Adult(split=Adult.Splits.RACE))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run_test(train, test)
    tprs = metric_per_sens(predictions, test, TPR())
    assert TPR().name == "TPR"
    test_dict = {
        "race_0": approx(0.37, abs=0.01),
        "race_1": approx(0.12, abs=0.01),
        "race_2": approx(0.14, abs=0.01),
        "race_3": approx(0.12, abs=0.01),
        "race_4": approx(0.16, abs=0.01),
    }

    for key, val in tprs.items():
        assert val == test_dict[key]

    tpr_diff = ratio_per_sens(tprs)
    test_dict = {
        "race_0÷race_1": approx(0.32, abs=0.1),
        "race_0÷race_2": approx(0.37, abs=0.1),
        "race_0÷race_3": approx(0.33, abs=0.1),
        "race_0÷race_4": approx(0.44, abs=0.1),
        "race_1÷race_2": approx(0.88, abs=0.1),
        "race_1÷race_3": approx(0.97, abs=0.1),
        "race_1÷race_4": approx(0.72, abs=0.1),
        "race_2÷race_3": approx(0.91, abs=0.1),
        "race_2÷race_4": approx(0.74, abs=0.1),
        "race_3÷race_4": approx(0.74, abs=0.1),
    }

    for key, val in tpr_diff.items():
        assert val == test_dict[key]


def test_nb_acc():
    """Test nb acc."""
    data: DataTuple = load_data(NonBinaryToy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run_test(train, test)
    acc_score = Accuracy().score(predictions, test)
    assert acc_score == 0.2625


def test_nb_tpr():
    """Test nb tpr."""
    data: DataTuple = load_data(NonBinaryToy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run_test(train, test)
    tpr_score = TPR(pos_class=0).score(predictions, test)
    assert tpr_score == 0.0
    tpr_score = TPR(pos_class=1).score(predictions, test)
    assert tpr_score == 0.0
    tpr_score = TPR(pos_class=2).score(predictions, test)
    assert tpr_score == 0.95
    tpr_score = TPR(pos_class=3).score(predictions, test)
    assert tpr_score == 0.14285714285714285
    tpr_score = TPR(pos_class=4).score(predictions, test)
    assert tpr_score == 0.0

    with pytest.raises(LabelOutOfBounds):
        _ = TPR(pos_class=5).score(predictions, test)

    accs = metric_per_sens(predictions, test, TPR())
    assert accs == {
        "sensitive-attr_1_0": approx(0.0, abs=0.1),
        "sensitive-attr_1_1": approx(0.0, abs=0.1),
    }

    model = LR()
    predictions = model.run_test(train, test)

    print([(k, z) for k, z in zip(predictions.hard.values, test.y.values) if k != z])

    tpr_score = TPR(pos_class=0).score(predictions, test)
    assert tpr_score == 0.0
    tpr_score = TPR(pos_class=1).score(predictions, test)
    assert tpr_score == 0.0
    tpr_score = TPR(pos_class=2).score(predictions, test)
    assert tpr_score == 0.45
    tpr_score = TPR(pos_class=3).score(predictions, test)
    assert tpr_score == 0.21428571428571427
    tpr_score = TPR(pos_class=4).score(predictions, test)
    assert tpr_score == 0.0

    with pytest.raises(LabelOutOfBounds):
        _ = TPR(pos_class=5).score(predictions, test)

    tprs = metric_per_sens(predictions, test, TPR())
    assert tprs == {
        "sensitive-attr_1_0": approx(0.0, abs=0.1),
        "sensitive-attr_1_1": approx(0.0, abs=0.1),
    }


def test_nb_tnr():
    """Test nb tnr."""
    data: DataTuple = load_data(NonBinaryToy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: Prediction = model.run_test(train, test)
    tnr_score = TNR(pos_class=0).score(predictions, test)
    assert tnr_score == 1.0
    tnr_score = TNR(pos_class=1).score(predictions, test)
    assert tnr_score == 1.0
    tnr_score = TNR(pos_class=2).score(predictions, test)
    assert tnr_score == approx(0.066, abs=0.01)
    tnr_score = TNR(pos_class=3).score(predictions, test)
    assert tnr_score == approx(0.954, abs=0.001)
    tnr_score = TNR(pos_class=4).score(predictions, test)
    assert tnr_score == 1.0

    with pytest.raises(LabelOutOfBounds):
        _ = TNR(pos_class=5).score(predictions, test)

    accs = metric_per_sens(predictions, test, TNR())
    assert accs == {
        "sensitive-attr_1_0": approx(1.0, abs=0.1),
        "sensitive-attr_1_1": approx(1.0, abs=0.1),
    }

    model = LR()
    predictions = model.run_test(train, test)

    print([(k, z) for k, z in zip(predictions.hard.values, test.y.values) if k != z])

    tnr_score = TNR(pos_class=0).score(predictions, test)
    assert tnr_score == approx(0.939, abs=0.01)
    tnr_score = TNR(pos_class=1).score(predictions, test)
    assert tnr_score == approx(0.9846, abs=0.01)
    tnr_score = TNR(pos_class=2).score(predictions, test)
    assert tnr_score == approx(0.3166, abs=0.01)
    tnr_score = TNR(pos_class=3).score(predictions, test)
    assert tnr_score == approx(0.85, abs=0.1)
    tnr_score = TNR(pos_class=4).score(predictions, test)
    assert tnr_score == approx(0.88, abs=0.01)

    with pytest.raises(LabelOutOfBounds):
        _ = TNR(pos_class=5).score(predictions, test)

    tnrs = metric_per_sens(predictions, test, TNR())
    assert tnrs == {
        "sensitive-attr_1_0": approx(1.0, abs=0.1),
        "sensitive-attr_1_1": approx(1.0, abs=0.1),
    }


def _compute_di(preds: Prediction, actual: EvalTuple) -> float:
    ratios = ratio_per_sens(metric_per_sens(preds, actual, ProbPos()))
    return next(iter(ratios.values()))


def _compute_inv_cv(preds: Prediction, actual: EvalTuple) -> float:
    diffs = diff_per_sens(metric_per_sens(preds, actual, ProbPos()))
    return next(iter(diffs.values()))


def test_dependence_measures(simple_data: DataTuple) -> None:
    """Test dependence measures."""
    train_percentage = 0.75
    unbalanced, balanced, _ = BalancedTestSplit(train_percentage=train_percentage)(simple_data)

    fair_prediction = Prediction(hard=balanced.y)  # predict the balanced label
    unfair_prediction = Prediction(hard=unbalanced.y)  # predict the normal label
    extremely_unfair_prediction = Prediction(hard=unbalanced.s)  # predict s

    # measure the dependence between s and the prediction in several ways
    assert _compute_di(fair_prediction, balanced) == approx(1, abs=1e-15)
    assert _compute_di(unfair_prediction, unbalanced) == approx(0.602, abs=3e-3)
    assert _compute_di(extremely_unfair_prediction, unbalanced) == approx(0, abs=3e-3)
    assert _compute_inv_cv(fair_prediction, balanced) == approx(0, abs=1e-15)
    assert _compute_inv_cv(unfair_prediction, unbalanced) == approx(0.265, abs=3e-3)
    assert _compute_inv_cv(extremely_unfair_prediction, unbalanced) == approx(1, abs=3e-3)
    nmi = NMI()
    assert nmi.score(fair_prediction, balanced) == approx(0, abs=1e-15)
    assert nmi.score(unfair_prediction, unbalanced) == approx(0.0437, abs=3e-4)
    assert nmi.score(extremely_unfair_prediction, unbalanced) == approx(1, abs=3e-4)
    yanovich = Yanovich()
    assert yanovich.score(fair_prediction, balanced) == approx(0, abs=1e-15)
    assert yanovich.score(unfair_prediction, unbalanced) == approx(0.0702, abs=3e-4)
    assert yanovich.score(extremely_unfair_prediction, unbalanced) == approx(1, abs=3e-4)
    renyi = RenyiCorrelation()
    assert renyi.score(fair_prediction, balanced) == approx(0, abs=1e-15)
    assert renyi.score(unfair_prediction, unbalanced) == approx(0.234, abs=3e-4)
    assert renyi.score(extremely_unfair_prediction, unbalanced) == approx(1, abs=3e-4)


def test_dependence_measures_adult() -> None:
    """Test dependence measures."""
    data = load_data(Adult(split=Adult.Splits.SEX))
    train_percentage = 0.75
    unbalanced, balanced, _ = BalancedTestSplit(train_percentage=train_percentage)(data)

    fair_prediction = Prediction(hard=balanced.y)  # predict the balanced label
    unfair_prediction = Prediction(hard=unbalanced.y)  # predict the normal label
    extremely_unfair_prediction = Prediction(hard=unbalanced.s)  # predict s

    # measure the dependence between s and the prediction in several ways
    assert _compute_di(fair_prediction, balanced) == approx(1, abs=1e-15)
    assert _compute_di(unfair_prediction, unbalanced) == approx(0.364, abs=3e-3)
    assert _compute_di(extremely_unfair_prediction, unbalanced) == approx(0, abs=3e-3)
    assert _compute_inv_cv(fair_prediction, balanced) == approx(0, abs=1e-15)
    assert _compute_inv_cv(unfair_prediction, unbalanced) == approx(0.199, abs=3e-3)
    assert _compute_inv_cv(extremely_unfair_prediction, unbalanced) == approx(1, abs=3e-3)
    nmi = NMI()
    assert nmi.score(fair_prediction, balanced) == approx(0, abs=1e-15)
    assert nmi.score(unfair_prediction, unbalanced) == approx(0.0432, abs=3e-4)
    assert nmi.score(extremely_unfair_prediction, unbalanced) == approx(1, abs=3e-4)
    yanovich = Yanovich()
    assert yanovich.score(fair_prediction, balanced) == approx(0, abs=1e-15)
    assert yanovich.score(unfair_prediction, unbalanced) == approx(0.0396, abs=3e-4)
    assert yanovich.score(extremely_unfair_prediction, unbalanced) == approx(1, abs=3e-4)
    renyi = RenyiCorrelation()
    assert renyi.score(fair_prediction, balanced) == approx(0, abs=1e-15)
    assert renyi.score(unfair_prediction, unbalanced) == approx(0.216, abs=3e-4)
    assert renyi.score(extremely_unfair_prediction, unbalanced) == approx(1, abs=3e-4)
