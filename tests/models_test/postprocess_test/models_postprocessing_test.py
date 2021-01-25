"""EthicML tests."""
import pytest

import ethicml as em
from ethicml import (
    LR,
    Hardt,
    InAlgorithm,
    PostAlgorithm,
    Prediction,
    ProbPos,
    TrainTestPair,
    diff_per_sensitive_attribute,
    metric_per_sensitive_attribute,
)
from ethicml.algorithms.postprocess.dp_flip import DPFlip
from ethicml.utility.data_structures import TrainValPair
from tests.run_algorithm_test import count_true


def test_dp_flip(toy_train_test: TrainValPair) -> None:
    """Test the dem par flipping method."""
    train, test = toy_train_test
    train_test = em.concat_tt([train, test], ignore_index=True)

    in_model: InAlgorithm = LR()
    assert in_model is not None
    assert in_model.name == "Logistic Regression (C=1.0)"

    predictions: Prediction = in_model.run(train, train_test)

    # seperate out predictions on train set and predictions on test set
    pred_train = predictions.hard.iloc[: train.y.shape[0]]
    pred_test = predictions.hard.iloc[train.y.shape[0] :].reset_index(drop=True)
    assert count_true(pred_test.values == 1) == 44
    assert count_true(pred_test.values == 0) == 36

    post_model: PostAlgorithm = DPFlip()
    assert post_model.name == "DemPar. Post Process"
    fair_preds = post_model.run(Prediction(pred_train), train, Prediction(pred_test), test)
    assert count_true(fair_preds.hard.values == 1) == 57
    assert count_true(fair_preds.hard.values == 0) == 23
    diffs = diff_per_sensitive_attribute(
        metric_per_sensitive_attribute(fair_preds, test, ProbPos())
    )
    for name, diff in diffs.items():
        assert 0 == pytest.approx(diff, abs=1e-2)


def test_dp_flip_inverted_s(toy_train_test: TrainValPair) -> None:
    """Test the dem par flipping method."""
    train, test = toy_train_test
    train = train.replace(s=1 - train.s)
    test = test.replace(s=1 - test.s)
    train_test = em.concat_tt([train, test], ignore_index=True)

    in_model: InAlgorithm = LR()
    assert in_model is not None
    assert in_model.name == "Logistic Regression (C=1.0)"

    predictions: Prediction = in_model.run(train, train_test)

    # seperate out predictions on train set and predictions on test set
    pred_train = predictions.hard.iloc[: train.y.shape[0]]
    pred_test = predictions.hard.iloc[train.y.shape[0] :].reset_index(drop=True)
    assert count_true(pred_test.values == 1) == 44
    assert count_true(pred_test.values == 0) == 36

    post_model: PostAlgorithm = DPFlip()
    assert post_model.name == "DemPar. Post Process"
    fair_preds = post_model.run(Prediction(pred_train), train, Prediction(pred_test), test)
    assert count_true(fair_preds.hard.values == 1) == 57
    assert count_true(fair_preds.hard.values == 0) == 23
    diffs = diff_per_sensitive_attribute(
        metric_per_sensitive_attribute(fair_preds, test, ProbPos())
    )
    for name, diff in diffs.items():
        assert 0 == pytest.approx(diff, abs=1e-2)


def test_hardt(toy_train_test: TrainTestPair) -> None:
    """Tests the hardt postprocessing technique.

    Args:
        toy_train_test: Train-test pair of toy data
    """
    train, test = toy_train_test
    train_test = em.concat_tt([train, test], ignore_index=True)

    in_model: InAlgorithm = LR()
    assert in_model is not None
    assert in_model.name == "Logistic Regression (C=1.0)"

    predictions: Prediction = in_model.run(train, train_test)

    # seperate out predictions on train set and predictions on test set
    pred_train = predictions.hard.iloc[: train.y.shape[0]]
    pred_test = predictions.hard.iloc[train.y.shape[0] :]
    assert count_true(pred_test.values == 1) == 44
    assert count_true(pred_test.values == 0) == 36

    post_model: PostAlgorithm = Hardt()
    fair_preds = post_model.run(Prediction(pred_train), train, Prediction(pred_test), test)
    assert count_true(fair_preds.hard.values == 1) == 35
    assert count_true(fair_preds.hard.values == 0) == 45
