from typing import Tuple
import pandas as pd

from ethicml.algorithms import run_blocking
from ethicml.algorithms.inprocess import InAlgorithm, SVM
from ethicml.algorithms.preprocess import PreAlgorithm, PreAlgorithmAsync, Beutel, Zemel, VFAE
from ethicml.algorithms.preprocess.upsampler import Upsampler
from ethicml.utility import DataTuple, FairType, TestTuple
from tests.run_algorithm_test import get_train_test


def test_beutel():
    train, test = get_train_test()

    beut_model: PreAlgorithm = Beutel()
    assert beut_model is not None
    assert beut_model.name == "Beutel"

    new_train_test: Tuple[DataTuple, TestTuple] = beut_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions: pd.DataFrame = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 201
    assert predictions.values[predictions.values == -1].shape[0] == 199


def test_vfae():
    train, test = get_train_test()

    vfae_model: PreAlgorithm = VFAE(dataset="Toy", epochs=10, batch_size=100)
    assert vfae_model is not None
    assert vfae_model.name == "VFAE"

    new_train_test: Tuple[DataTuple, TestTuple] = vfae_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions: pd.DataFrame = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 201
    assert predictions.values[predictions.values == -1].shape[0] == 199

    vfae_model = VFAE(dataset="Toy", epochs=10, fairness="Eq. Opp", batch_size=100)
    assert vfae_model is not None
    assert vfae_model.name == "VFAE"

    new_train_test = vfae_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 201
    assert predictions.values[predictions.values == -1].shape[0] == 199

    vfae_model = VFAE(dataset="Toy", supervised=False, epochs=10,
                      fairness="Eq. Opp", batch_size=100)
    assert vfae_model is not None
    assert vfae_model.name == "VFAE"

    new_train_test = vfae_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 207
    assert predictions.values[predictions.values == -1].shape[0] == 193


def test_zemel():
    train, test = get_train_test()

    zemel_model: PreAlgorithm = Zemel()
    assert zemel_model is not None
    assert zemel_model.name == "Zemel"

    new_train_test: Tuple[DataTuple, TestTuple] = zemel_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions: pd.DataFrame = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 182
    assert predictions.values[predictions.values == -1].shape[0] == 218


def test_threaded_zemel():
    train, test = get_train_test()

    model: PreAlgorithmAsync = Zemel()
    assert model is not None
    assert model.name == "Zemel"

    new_train_test: Tuple[DataTuple, TestTuple] = run_blocking(
        model.run_async(train, test)
    )
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: pd.DataFrame = classifier.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 182
    assert predictions.values[predictions.values == -1].shape[0] == 218

    beut_model: PreAlgorithm = Zemel()
    assert beut_model is not None
    assert beut_model.name == "Zemel"

    new_train_test = beut_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 182
    assert predictions.values[predictions.values == -1].shape[0] == 218


def test_threaded_beutel():
    train, test = get_train_test()

    model: PreAlgorithmAsync = Beutel()
    assert model is not None
    assert model.name == "Beutel"

    new_train_test: Tuple[DataTuple, TestTuple] = run_blocking(
        model.run_async(train, test)
    )
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: pd.DataFrame = classifier.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 201
    assert predictions.values[predictions.values == -1].shape[0] == 199

    beut_model: PreAlgorithm = Beutel()
    assert beut_model is not None
    assert beut_model.name == "Beutel"

    new_train_test = beut_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 201
    assert predictions.values[predictions.values == -1].shape[0] == 199


def test_threaded_vfae():
    train, test = get_train_test()

    model: PreAlgorithmAsync = VFAE(dataset='Toy')
    assert model is not None
    assert model.name == "VFAE"

    new_train_test: Tuple[DataTuple, TestTuple] = run_blocking(
        model.run_async(train, test)
    )
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: pd.DataFrame = classifier.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 193
    assert predictions.values[predictions.values == -1].shape[0] == 207


def test_threaded_custom_beutel():
    train, test = get_train_test()

    beut_model: PreAlgorithm = Beutel(epochs=5, fairness=FairType.EOPP)
    assert beut_model is not None
    assert beut_model.name == "Beutel"

    new_train_test_non_thread: Tuple[DataTuple, TestTuple] = beut_model.run(train, test)
    new_train_nt, new_test_nt = new_train_test_non_thread

    assert new_train_nt.x.shape[0] == train.x.shape[0]
    assert new_test_nt.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train_nt, new_test_nt)
    assert predictions.values[predictions.values == 1].shape[0] == 202
    assert predictions.values[predictions.values == -1].shape[0] == 198

    model: PreAlgorithmAsync = Beutel(epochs=5, fairness=FairType.EOPP)
    assert model is not None
    assert model.name == "Beutel"

    new_train_test: Tuple[DataTuple, TestTuple] = run_blocking(
        model.run_async(train, test)
    )
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    treaded_predictions: pd.DataFrame = classifier.run_test(new_train, new_test)
    assert treaded_predictions.values[treaded_predictions.values == 1].shape[0] == 202
    assert treaded_predictions.values[treaded_predictions.values == -1].shape[0] == 198


def test_upsampler():
    train, test = get_train_test()

    upsampler: PreAlgorithm = Upsampler()
    assert upsampler is not None
    assert upsampler.name == "Upsample"

    new_train: DataTuple
    new_test: TestTuple
    new_train, new_test = upsampler.run(train, test)

    assert new_test.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 218
    assert predictions.values[predictions.values == -1].shape[0] == 182


def test_async_upsampler():
    train, test = get_train_test()

    upsampler: PreAlgorithmAsync = Upsampler()
    assert upsampler is not None
    assert upsampler.name == "Upsample"

    new_train: DataTuple
    new_test: TestTuple
    new_train, new_test = run_blocking(
        upsampler.run_async(train, test)
    )

    assert new_test.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.values[predictions.values == 1].shape[0] == 218
    assert predictions.values[predictions.values == -1].shape[0] == 182
