"""Tests for hreaded models."""
from typing import NamedTuple

import pytest

import ethicml as em
from ethicml import InAlgorithmAsync, Prediction, TrainTestPair
from tests.conftest import get_id


class ThreadedParams(NamedTuple):
    """Define a test for a metric applied per sensitive group."""

    model: InAlgorithmAsync
    name: str
    num_pos: int


# THREADED_PARAMS = [ThreadedParams(model=SVMAsync(dir="/tmp"), name="SVM", num_pos=45)]


# @pytest.mark.parametrize("model,name,num_pos", THREADED_PARAMS, ids=get_id)
# def test_threaded(toy_train_test: TrainTestPair, model: InAlgorithmAsync, name: str, num_pos: int):
#     """Test threaded svm."""
#     train, test = toy_train_test

#     assert model is not None
#     assert model.name == name

#     predictions: Prediction = model.run(train, test)
#     assert predictions.hard.values[predictions.hard.values == 1].shape[0] == num_pos
#     num_neg = predictions.hard.values[predictions.hard.values == 0].shape[0]
#     assert num_neg == len(predictions) - num_pos


# @pytest.mark.parametrize("model,name,num_pos", THREADED_PARAMS, ids=get_id)
# def test_threaded_sep(
#     toy_train_test: TrainTestPair, model: InAlgorithmAsync, name: str, num_pos: int
# ):
#     """Test threaded svm."""
#     train, test = toy_train_test

#     assert model is not None
#     assert model.name == name

#     model = model.fit(train)
#     predictions: Prediction = model.predict(test)
#     assert predictions.hard.values[predictions.hard.values == 1].shape[0] == num_pos
#     num_neg = predictions.hard.values[predictions.hard.values == 0].shape[0]
#     assert num_neg == len(predictions) - num_pos
