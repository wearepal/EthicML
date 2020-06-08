"""EthicML tests"""

from ethicml.algorithms.inprocess import LR, InAlgorithm
from ethicml.algorithms.postprocess import Hardt, PostAlgorithm
from ethicml.utility import Prediction, TrainTestPair, concat_tt
from tests.run_algorithm_test import count_true


def test_hardt(toy_train_test: TrainTestPair) -> None:
    """
    tests the hardt postprocessing technique
    Args:
        toy_train_test: Train-test pair of toy data
    """
    train, test = toy_train_test
    train_test = concat_tt([train, test], ignore_index=True)

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
