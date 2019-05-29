import asyncio
from typing import Tuple
import pandas as pd

from ethicml.algorithms.inprocess import InAlgorithm, LR, LRProb, SVM
from ethicml.algorithms.preprocess import PreAlgorithm, Beutel, Zemel
from ethicml.algorithms.preprocess.vfae import VFAE
from ethicml.algorithms.utils import DataTuple
from tests.run_algorithm_test import get_train_test


def test_beutel():
    train, test = get_train_test()

    beut_model: PreAlgorithm = Beutel()
    assert beut_model is not None
    assert beut_model.name == "Beutel"

    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = beut_model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions: pd.DataFrame = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 208
    assert predictions.values[predictions.values == -1].shape[0] == 192


def test_vfae():
    train, test = get_train_test()

    vfae_model: PreAlgorithm = VFAE(dataset="Toy", epochs=10, batch_size=100)
    assert vfae_model is not None
    assert vfae_model.name == "VFAE"

    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = vfae_model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions: pd.DataFrame = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 215
    assert predictions.values[predictions.values == -1].shape[0] == 185

    vfae_model = VFAE(dataset="Toy", epochs=10, fairness="Eq. Opp", batch_size=100)
    assert vfae_model is not None
    assert vfae_model.name == "VFAE"

    new_xtrain_xtest = vfae_model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 215
    assert predictions.values[predictions.values == -1].shape[0] == 185

    vfae_model = VFAE(dataset="Toy", supervised=False, epochs=10,
                      fairness="Eq. Opp", batch_size=100)
    assert vfae_model is not None
    assert vfae_model.name == "VFAE"

    new_xtrain_xtest = vfae_model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 205
    assert predictions.values[predictions.values == -1].shape[0] == 195


def test_zemel():
    train, test = get_train_test()

    zemel_model: PreAlgorithm = Zemel()
    assert zemel_model is not None
    assert zemel_model.name == "Zemel"

    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = zemel_model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions: pd.DataFrame = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 182
    assert predictions.values[predictions.values == -1].shape[0] == 218


def test_threaded_zemel():
    train, test = get_train_test()

    model: PreAlgorithm = Zemel()
    assert model is not None
    assert model.name == "Zemel"

    loop = asyncio.get_event_loop()
    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = loop.run_until_complete(
        model.run_async(train, test)
    )
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: pd.DataFrame = classifier.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 182
    assert predictions.values[predictions.values == -1].shape[0] == 218

    beut_model: PreAlgorithm = Zemel()
    assert beut_model is not None
    assert beut_model.name == "Zemel"

    new_xtrain_xtest = beut_model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 182
    assert predictions.values[predictions.values == -1].shape[0] == 218


def test_threaded_beutel():
    train, test = get_train_test()

    model: PreAlgorithm = Beutel()
    assert model is not None
    assert model.name == "Beutel"

    loop = asyncio.get_event_loop()
    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = loop.run_until_complete(
        model.run_async(train, test)
    )
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: pd.DataFrame = classifier.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 208
    assert predictions.values[predictions.values == -1].shape[0] == 192

    beut_model: PreAlgorithm = Beutel()
    assert beut_model is not None
    assert beut_model.name == "Beutel"

    new_xtrain_xtest = beut_model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 208
    assert predictions.values[predictions.values == -1].shape[0] == 192


def test_threaded_custom_beutel():
    train, test = get_train_test()

    beut_model: PreAlgorithm = Beutel(epochs=5, fairness="Eq. Opp")
    assert beut_model is not None
    assert beut_model.name == "Beutel"

    new_xtrain_xtest_non_thread: Tuple[pd.DataFrame, pd.DataFrame] = beut_model.run(train, test)
    new_xtrain_nt, new_xtest_nt = new_xtrain_xtest_non_thread

    assert new_xtrain_nt.shape[0] == train.x.shape[0]
    assert new_xtest_nt.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain_nt, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest_nt, s=test.s, y=test.y)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 202
    assert predictions.values[predictions.values == -1].shape[0] == 198

    model: PreAlgorithm = Beutel(epochs=5, fairness="Eq. Opp")
    assert model is not None
    assert model.name == "Beutel"

    loop = asyncio.get_event_loop()
    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = loop.run_until_complete(
        model.run_async(train, test)
    )
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    treaded_predictions: pd.DataFrame = classifier.run_test(new_train, new_test)
    assert treaded_predictions.values[treaded_predictions.values == 1].shape[0] == 202
    assert treaded_predictions.values[treaded_predictions.values == -1].shape[0] == 198
