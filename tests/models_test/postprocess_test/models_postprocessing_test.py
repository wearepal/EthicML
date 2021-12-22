"""EthicML tests."""
from typing import NamedTuple

import pytest

import ethicml as em
from ethicml import LR, Hardt, InAlgorithm, PostAlgorithm, Prediction, ProbPos
from ethicml.algorithms.postprocess.dp_flip import DPFlip
from ethicml.utility.data_structures import TrainValPair
from tests.run_algorithm_test import count_true


class PostprocessTest(NamedTuple):
    """Define a test for an postprocess model."""

    post_model: PostAlgorithm
    name: str
    num_pos: int


@pytest.mark.parametrize(
    "post_model,name,num_pos",
    [
        PostprocessTest(post_model=DPFlip(), name="DemPar. Post Process", num_pos=57),
        PostprocessTest(post_model=Hardt(), name="Hardt", num_pos=35),
    ],
)
def test_post(
    toy_train_test: TrainValPair, post_model: PostAlgorithm, name: str, num_pos: int
) -> None:
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

    assert post_model.name == name
    fair_preds = post_model.run(Prediction(pred_train), train, Prediction(pred_test), test)
    assert count_true(fair_preds.hard.values == 1) == num_pos
    assert count_true(fair_preds.hard.values == 0) == len(fair_preds) - num_pos
    diffs = em.diff_per_sensitive_attribute(
        em.metric_per_sensitive_attribute(fair_preds, test, ProbPos())
    )
    if isinstance(post_model, DPFlip):
        for diff in diffs.values():
            assert pytest.approx(diff, abs=1e-2) == 0


@pytest.mark.parametrize(
    "post_model,name,num_pos",
    [
        PostprocessTest(post_model=DPFlip(), name="DemPar. Post Process", num_pos=57),
        PostprocessTest(post_model=Hardt(), name="Hardt", num_pos=35),
    ],
)
def test_post_sep_fit_pred(
    toy_train_test: TrainValPair, post_model: PostAlgorithm, name: str, num_pos: int
) -> None:
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

    assert post_model.name == name
    fair_model = post_model.fit(Prediction(pred_train), train)
    fair_preds = fair_model.predict(Prediction(pred_test), test)
    assert count_true(fair_preds.hard.values == 1) == num_pos
    assert count_true(fair_preds.hard.values == 0) == len(fair_preds) - num_pos
    diffs = em.diff_per_sensitive_attribute(
        em.metric_per_sensitive_attribute(fair_preds, test, ProbPos())
    )
    if isinstance(post_model, DPFlip):
        for diff in diffs.values():
            assert pytest.approx(diff, abs=1e-2) == 0


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
    diffs = em.diff_per_sensitive_attribute(
        em.metric_per_sensitive_attribute(fair_preds, test, ProbPos())
    )
    for diff in diffs.values():
        assert pytest.approx(diff, abs=1e-2) == 0
