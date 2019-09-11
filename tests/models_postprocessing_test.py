"""EthicML tests"""


import pandas as pd

from ethicml.algorithms.inprocess import InAlgorithm, LR
from ethicml.algorithms.postprocess import Hardt, PostAlgorithm
from ethicml.utility import TrainTestPair, concat_tt


def test_hardt(toy_train_test: TrainTestPair) -> None:
    """
    tests the hardt postprocessing technique
    Args:
        toy_train_test:

    Returns:

    """
    train, test = toy_train_test
    train_test = concat_tt([train, test], ignore_index=True)

    in_model: InAlgorithm = LR()
    assert in_model is not None
    assert in_model.name == "Logistic Regression, C=1.0"

    predictions: pd.DataFrame = in_model.run(train, train_test)

    # seperate out predictions on train set and predictions on test set
    pred_train = predictions.iloc[: train.y.shape[0]]
    pred_test = predictions.iloc[train.y.shape[0] :]
    assert (pred_test.values == 1).sum() == 211
    assert (pred_test.values == -1).sum() == 189

    post_model: PostAlgorithm = Hardt(-1, 1)
    fair_preds = post_model.run(pred_train, train, pred_test, test)
    assert (fair_preds.values == 1).sum() == 111
    assert (fair_preds.values == -1).sum() == 289
