import pandas as pd
from typing import Tuple

from ethicml.algorithms.inprocess import InAlgorithm, LR
from ethicml.algorithms.preprocess import PreAlgorithm
from ethicml.algorithms.preprocess.imagine import Imagine
from ethicml.data import load_data, Toy
from ethicml.preprocessing import train_test_split
from ethicml.utility import DataTuple, TestTuple


def main():
    data: DataTuple = load_data(Toy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test

    imaginer: PreAlgorithm = Imagine(epochs=200, batch_size=32)
    assert imaginer is not None
    assert imaginer.name == "Imagined Examples"

    new_train: DataTuple
    new_test: TestTuple
    new_train, new_test = imaginer.run(train, test)

    assert new_test.x.shape[0] == test.x.shape[0]
    # assert new_test.name == test.name
    # assert new_train.name == train.name

    lr_model: InAlgorithm = LR()
    assert lr_model is not None
    assert lr_model.name == "Logistic Regression"

    pd.testing.assert_frame_equal(train.y, new_train.y)

    predictions = lr_model.run_test(new_train, test)
    predictions_og = lr_model.run_test(train, test)
    pd.testing.assert_frame_equal(predictions, predictions_og)
    assert predictions.values[predictions.values == 1].shape[0] == \
           predictions_og.values[predictions_og.values == 1].shape[0]
    assert predictions.values[predictions.values == -1].shape[0] == \
           predictions_og.values[predictions_og.values == -1].shape[0]


if __name__ == '__main__':
    main()
