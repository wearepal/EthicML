"""EthicML tests."""
from typing import NamedTuple

import numpy as np
import pytest

import ethicml as em
from ethicml import Prediction, TrainValPair
from ethicml.metrics import ProbPos, diff_per_sens, metric_per_sens
from ethicml.models import LR, DPFlip, Hardt, InAlgorithm, PostAlgorithm


class PostprocessTest(NamedTuple):
    """Define a test for an postprocess model."""

    post_model: PostAlgorithm
    name: str
    num_pos: int


@pytest.mark.parametrize(
    ("post_model", "name", "num_pos"),
    [
        PostprocessTest(post_model=DPFlip(), name="DemPar. Post Process", num_pos=57),
        PostprocessTest(post_model=Hardt(), name="Hardt", num_pos=35),
    ],
)
def test_post(
    toy_train_val: TrainValPair, post_model: PostAlgorithm, name: str, num_pos: int
) -> None:
    """Test the dem par flipping method."""
    train, test = toy_train_val
    train_test = em.concat([train, test], ignore_index=True)

    in_model: InAlgorithm = LR()
    assert in_model is not None
    assert in_model.name == "Logistic Regression (C=1.0)"

    predictions: Prediction = in_model.run(train, train_test)

    # seperate out predictions on train set and predictions on test set
    pred_train = predictions.hard.iloc[: train.y.shape[0]]
    pred_test = predictions.hard.iloc[train.y.shape[0] :].reset_index(drop=True)
    assert np.count_nonzero(pred_test.to_numpy() == 1) == 44
    assert np.count_nonzero(pred_test.to_numpy() == 0) == 36

    assert post_model.name == name
    fair_preds = post_model.run(Prediction(pred_train), train, Prediction(pred_test), test)
    assert np.count_nonzero(fair_preds.hard.to_numpy() == 1) == num_pos
    assert np.count_nonzero(fair_preds.hard.to_numpy() == 0) == len(fair_preds) - num_pos
    diffs = diff_per_sens(metric_per_sens(fair_preds, test, ProbPos()))
    if isinstance(post_model, DPFlip):
        for diff in diffs.values():
            assert pytest.approx(diff, abs=1e-2) == 0


@pytest.mark.parametrize(
    ("post_model", "name", "num_pos"),
    [
        PostprocessTest(post_model=DPFlip(), name="DemPar. Post Process", num_pos=57),
        PostprocessTest(post_model=Hardt(), name="Hardt", num_pos=35),
    ],
)
@pytest.mark.xdist_group("post_model_files")
def test_post_sep_fit_pred(
    toy_train_val: TrainValPair, post_model: PostAlgorithm, name: str, num_pos: int
) -> None:
    """Test the dem par flipping method."""
    train, test = toy_train_val
    train_test = em.concat([train, test], ignore_index=True)

    in_model: InAlgorithm = LR()
    assert in_model is not None
    assert in_model.name == "Logistic Regression (C=1.0)"

    predictions: Prediction = in_model.run(train, train_test)

    # seperate out predictions on train set and predictions on test set
    pred_train = predictions.hard.iloc[: train.y.shape[0]]
    pred_test = predictions.hard.iloc[train.y.shape[0] :].reset_index(drop=True)
    assert np.count_nonzero(pred_test.to_numpy() == 1) == 44
    assert np.count_nonzero(pred_test.to_numpy() == 0) == 36

    assert post_model.name == name
    fair_model = post_model.fit(Prediction(pred_train), train)
    fair_preds = fair_model.predict(Prediction(pred_test), test)
    assert np.count_nonzero(fair_preds.hard.to_numpy() == 1) == num_pos
    assert np.count_nonzero(fair_preds.hard.to_numpy() == 0) == len(fair_preds) - num_pos
    diffs = diff_per_sens(metric_per_sens(fair_preds, test, ProbPos()))
    if isinstance(post_model, DPFlip):
        for diff in diffs.values():
            assert pytest.approx(diff, abs=1e-2) == 0


def test_dp_flip_inverted_s(toy_train_val: TrainValPair) -> None:
    """Test the dem par flipping method."""
    train, test = toy_train_val
    train = train.replace(s=1 - train.s)
    test = test.replace(s=1 - test.s)
    train_test = em.concat([train, test], ignore_index=True)

    in_model: InAlgorithm = LR()
    assert in_model is not None
    assert in_model.name == "Logistic Regression (C=1.0)"

    predictions: Prediction = in_model.run(train, train_test)

    # seperate out predictions on train set and predictions on test set
    pred_train = predictions.hard.iloc[: train.y.shape[0]]
    pred_test = predictions.hard.iloc[train.y.shape[0] :].reset_index(drop=True)
    assert np.count_nonzero(pred_test.to_numpy() == 1) == 44
    assert np.count_nonzero(pred_test.to_numpy() == 0) == 36

    post_model: PostAlgorithm = DPFlip()
    assert post_model.name == "DemPar. Post Process"
    fair_preds = post_model.run(Prediction(pred_train), train, Prediction(pred_test), test)
    assert np.count_nonzero(fair_preds.hard.to_numpy() == 1) == 57
    assert np.count_nonzero(fair_preds.hard.to_numpy() == 0) == 23
    diffs = diff_per_sens(metric_per_sens(fair_preds, test, ProbPos()))
    for diff in diffs.values():
        assert pytest.approx(diff, abs=1e-2) == 0
