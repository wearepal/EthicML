"""
Test that we can get some metrics on predictions
"""

from typing import Tuple
import pandas as pd
import pytest
from pytest import approx

from ethicml.algorithms.inprocess import InAlgorithm, LRProb, SVM, LR, Kamiran
from ethicml.utility import DataTuple
from ethicml.data import Adult, NonBinaryToy, load_data
from ethicml.evaluators import (
    diff_per_sensitive_attribute,
    metric_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
    run_metrics,
    MetricNotApplicable,
)
from ethicml.metrics import (Accuracy, BCR, CV, Metric, NMI, PPV, NPV, ProbNeg,
                             ProbOutcome, ProbPos, TNR, TPR, Theil, Hsic, LabelOutOfBounds)
from ethicml.preprocessing import train_test_split
from tests.run_algorithm_test import get_train_test

RTOL = 1e-5  # relative tolerance when comparing two floats


def test_get_acc_of_predictions():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    acc: Metric = Accuracy()
    assert acc.name == "Accuracy"
    score = acc.score(predictions, test)
    assert score == 0.89


def test_mni_preds_and_s():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    acc: Metric = NMI(base='s')
    assert acc.name == "NMI preds and s"
    score = acc.score(predictions, test)
    assert score == pytest.approx(0.083, abs=0.001)


def test_accuracy_per_sens_attr():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, Accuracy())
    assert acc_per_sens == {'s_0': 0.905, 's_1': 0.875}


def test_probpos_per_sens_attr():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, ProbPos())
    assert acc_per_sens == {'s_0': 0.335, 's_1': 0.67}


def test_proboutcome_per_sens_attr():
    train, test = get_train_test()
    model: InAlgorithm = LRProb()
    predictions: pd.DataFrame = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, ProbOutcome())
    assert acc_per_sens == {'s_0': pytest.approx(0.372, abs=0.001),
                            's_1': pytest.approx(0.661, abs=0.001)}


def test_probneg_per_sens_attr():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, ProbNeg())
    assert acc_per_sens == {'s_0': 0.665, 's_1': 0.33}


def test_acc_per_nonbinary_sens():
    data: DataTuple = load_data(Adult("Nationality"))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, Accuracy())

    test_dict = {'native-country_Cambodia_0':    pytest.approx(0.778, abs=0.001),
                 'native-country_Cambodia_1':    pytest.approx(0.500, abs=0.001),
                 'native-country_Canada_0':      pytest.approx(0.779, abs=0.001),
                 'native-country_Canada_1':      pytest.approx(0.595, abs=0.001),
                 'native-country_China_0':       pytest.approx(0.778, abs=0.001),
                 'native-country_China_1':       pytest.approx(0.800, abs=0.001),
                 'native-country_Columbia_0':    pytest.approx(0.778, abs=0.001),
                 'native-country_Columbia_1':    pytest.approx(1.000, abs=0.001),
                 'native-country_Cuba_0':        pytest.approx(0.778, abs=0.001),
                 'native-country_Cuba_1':        pytest.approx(0.826, abs=0.001),
                 'native-country_Dominican-Republic_0': pytest.approx(0.778, abs=0.001),
                 'native-country_Dominican-Republic_1': pytest.approx(0.933, abs=0.001),
                 'native-country_Ecuador_0':     pytest.approx(0.778, abs=0.001),
                 'native-country_Ecuador_1':     pytest.approx(0.800, abs=0.001),
                 'native-country_El-Salvador_0': pytest.approx(0.778, abs=0.001),
                 'native-country_El-Salvador_1': pytest.approx(1.000, abs=0.001),
                 'native-country_England_0':     pytest.approx(0.779, abs=0.001),
                 'native-country_England_1':     pytest.approx(0.600, abs=0.001),
                 'native-country_France_0':      pytest.approx(0.778, abs=0.001),
                 'native-country_France_1':      pytest.approx(0.500, abs=0.001),
                 'native-country_Germany_0':     pytest.approx(0.778, abs=0.001),
                 'native-country_Germany_1':     pytest.approx(0.679, abs=0.001),
                 'native-country_Greece_0':      pytest.approx(0.778, abs=0.001),
                 'native-country_Greece_1':      pytest.approx(0.636, abs=0.001),
                 'native-country_Guatemala_0':   pytest.approx(0.778, abs=0.001),
                 'native-country_Guatemala_1':   pytest.approx(0.952, abs=0.001),
                 'native-country_Haiti_0':       pytest.approx(0.778, abs=0.001),
                 'native-country_Haiti_1':       pytest.approx(0.929, abs=0.001),
                 'native-country_Holand-Netherlands_0': pytest.approx(0.778, abs=0.001),
                 'native-country_Honduras_0':    pytest.approx(0.778, abs=0.001),
                 'native-country_Honduras_1':    pytest.approx(1.000, abs=0.001),
                 'native-country_Hong_0':        pytest.approx(0.778, abs=0.001),
                 'native-country_Hong_1':        pytest.approx(0.667, abs=0.001),
                 'native-country_Hungary_0':     pytest.approx(0.778, abs=0.001),
                 'native-country_Hungary_1':     pytest.approx(1.000, abs=0.001),
                 'native-country_India_0':       pytest.approx(0.778, abs=0.001),
                 'native-country_India_1':       pytest.approx(0.589, abs=0.001),
                 'native-country_Iran_0':        pytest.approx(0.778, abs=0.001),
                 'native-country_Iran_1':        pytest.approx(0.364, abs=0.001),
                 'native-country_Ireland_0':     pytest.approx(0.778, abs=0.001),
                 'native-country_Ireland_1':     pytest.approx(0.818, abs=0.001),
                 'native-country_Italy_0':       pytest.approx(0.778, abs=0.001),
                 'native-country_Italy_1':       pytest.approx(0.565, abs=0.001),
                 'native-country_Jamaica_0':     pytest.approx(0.778, abs=0.001),
                 'native-country_Jamaica_1':     pytest.approx(0.800, abs=0.001),
                 'native-country_Japan_0':       pytest.approx(0.778, abs=0.001),
                 'native-country_Japan_1':       pytest.approx(0.760, abs=0.001),
                 'native-country_Laos_0':        pytest.approx(0.778, abs=0.001),
                 'native-country_Laos_1':        pytest.approx(1.000, abs=0.001),
                 'native-country_Mexico_0':      pytest.approx(0.774, abs=0.001),
                 'native-country_Mexico_1':      pytest.approx(0.963, abs=0.001),
                 'native-country_Nicaragua_0':   pytest.approx(0.778, abs=0.001),
                 'native-country_Nicaragua_1':   pytest.approx(0.875, abs=0.001),
                 'native-country_Outlying-US(Guam-USVI-etc)_0': pytest.approx(0.778, abs=0.001),
                 'native-country_Outlying-US(Guam-USVI-etc)_1': pytest.approx(0.800, abs=0.001),
                 'native-country_Peru_0':        pytest.approx(0.778, abs=0.001),
                 'native-country_Peru_1':        pytest.approx(0.923, abs=0.001),
                 'native-country_Philippines_0': pytest.approx(0.778, abs=0.001),
                 'native-country_Philippines_1': pytest.approx(0.717, abs=0.001),
                 'native-country_Poland_0':      pytest.approx(0.778, abs=0.001),
                 'native-country_Poland_1':      pytest.approx(0.813, abs=0.001),
                 'native-country_Portugal_0':    pytest.approx(0.778, abs=0.001),
                 'native-country_Portugal_1':    pytest.approx(0.800, abs=0.001),
                 'native-country_Puerto-Rico_0': pytest.approx(0.778, abs=0.001),
                 'native-country_Puerto-Rico_1': pytest.approx(0.921, abs=0.001),
                 'native-country_Scotland_0':    pytest.approx(0.778, abs=0.001),
                 'native-country_Scotland_1':    pytest.approx(1.000, abs=0.001),
                 'native-country_South_0':       pytest.approx(0.778, abs=0.001),
                 'native-country_South_1':       pytest.approx(0.714, abs=0.001),
                 'native-country_Taiwan_0':      pytest.approx(0.778, abs=0.001),
                 'native-country_Taiwan_1':      pytest.approx(0.429, abs=0.001),
                 'native-country_Thailand_0':    pytest.approx(0.778, abs=0.001),
                 'native-country_Thailand_1':    pytest.approx(0.750, abs=0.001),
                 'native-country_Trinadad&Tobago_0': pytest.approx(0.778, abs=0.001),
                 'native-country_Trinadad&Tobago_1': pytest.approx(1.000, abs=0.001),
                 'native-country_United-States_0': pytest.approx(0.806, abs=0.001),
                 'native-country_United-States_1': pytest.approx(0.776, abs=0.001),
                 'native-country_Vietnam_0':     pytest.approx(0.778, abs=0.001),
                 'native-country_Vietnam_1':     pytest.approx(0.929, abs=0.001),
                 'native-country_Yugoslavia_0':  pytest.approx(0.778, abs=0.001),
                 'native-country_Yugoslavia_1':  pytest.approx(0.714, abs=0.001)}

    for k, v in acc_per_sens.items():
        assert acc_per_sens[k] == test_dict[k]


def test_acc_per_race():
    data: DataTuple = load_data(Adult("Race"))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, Accuracy())
    test_dict = {'race_Amer-Indian-Eskimo_0': pytest.approx(0.78, abs=0.01),
                 'race_Amer-Indian-Eskimo_1': pytest.approx(0.94, abs=0.01),
                 'race_Asian-Pac-Islander_0': pytest.approx(0.78, abs=0.01),
                 'race_Asian-Pac-Islander_1': pytest.approx(0.73, abs=0.01),
                 'race_Black_0': pytest.approx(0.77, abs=0.01),
                 'race_Black_1': pytest.approx(0.87, abs=0.01),
                 'race_Other_0': pytest.approx(0.78, abs=0.01),
                 'race_Other_1': pytest.approx(0.88, abs=0.01),
                 'race_White_0': pytest.approx(0.85, abs=0.01),
                 'race_White_1': pytest.approx(0.77, abs=0.01)}

    for k, v in acc_per_sens.items():
        assert acc_per_sens[k] == test_dict[k]


def test_tpr_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert TPR().name == "TPR"
    assert tprs == {'s_0': pytest.approx(0.84, abs=0.01), 's_1': pytest.approx(0.88, abs=0.01)}
    tpr_diff = diff_per_sensitive_attribute(tprs)
    print(tpr_diff)
    assert tpr_diff["s_0-s_1"] == pytest.approx(0.04, abs=0.01)


def test_get_nmi_of_predictions():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    nmi: Metric = NMI()
    assert nmi.name == "NMI preds and y"
    score = nmi.score(predictions, test)
    assert score == pytest.approx(0.50, abs=0.01)


def test_nmi_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    nmis = metric_per_sensitive_attribute(predictions, test, NMI())
    assert NMI().name == "NMI preds and y"
    assert nmis == {'s_0': pytest.approx(0.52, abs=0.01), 's_1': pytest.approx(0.42, abs=0.01)}
    nmi_diff = diff_per_sensitive_attribute(nmis)
    assert nmi_diff["s_0-s_1"] == pytest.approx(0.10, abs=0.01)


def test_ppv_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    results = metric_per_sensitive_attribute(predictions, test, PPV())
    assert PPV().name == "PPV"
    assert results == {'s_0': pytest.approx(0.88, abs=0.01), 's_1': pytest.approx(0.93, abs=0.01)}
    diff = diff_per_sensitive_attribute(results)
    assert diff["s_0-s_1"] == pytest.approx(0.05, abs=0.1)


def test_npv_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    results = metric_per_sensitive_attribute(predictions, test, NPV())
    assert NPV().name == "NPV"
    assert results == {'s_0': pytest.approx(0.91, abs=0.01), 's_1': pytest.approx(0.75, abs=0.01)}
    diff = diff_per_sensitive_attribute(results)
    assert diff["s_0-s_1"] == pytest.approx(0.15, abs=0.01)


def test_bcr_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    results = metric_per_sensitive_attribute(predictions, test, BCR())
    assert BCR().name == "BCR"
    assert results == {'s_0': pytest.approx(0.89, abs=0.01), 's_1': pytest.approx(0.86, abs=0.01)}
    diff = diff_per_sensitive_attribute(results)
    assert diff["s_0-s_1"] == pytest.approx(0.02, abs=0.01)


def test_cv():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    cv = CV()
    score = cv.score(predictions, test)
    assert CV().name == "CV"
    assert score == 0.665


def test_theil():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    theil = Theil()
    score = theil.score(predictions, test)
    assert Theil().name == "Theil_Index"
    assert score == pytest.approx(0.08, abs=0.01)

    model = LR()
    predictions = model.run(train, test)
    theil = Theil()
    score = theil.score(predictions, test)
    assert score == pytest.approx(0.07, abs=0.01)

    model = Kamiran()
    predictions = model.run(train, test)
    theil = Theil()
    score = theil.score(predictions, test)
    assert score == pytest.approx(0.08, abs=0.01)


def test_theil_per_sens_attr():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    theil_per_sens = metric_per_sensitive_attribute(predictions, test, Theil())
    assert theil_per_sens == {'s_0': pytest.approx(0.07, abs=0.01), 's_1': pytest.approx(0.10, abs=0.01)}


def test_hsic():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    hsic = Hsic()
    score = hsic.score(predictions, test)
    assert Hsic().name == "HSIC"
    assert score == pytest.approx(0.02, abs=0.01)


def test_use_appropriate_metric():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    with pytest.raises(MetricNotApplicable):
        metric_per_sensitive_attribute(predictions, test, CV())


def test_tnr_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    nmis = metric_per_sensitive_attribute(predictions, test, TNR())
    assert NMI().name == "NMI preds and y"
    assert nmis == {'s_0': pytest.approx(0.93, abs=0.01), 's_1': pytest.approx(0.84, abs=0.01)}
    nmi_diff = diff_per_sensitive_attribute(nmis)
    assert nmi_diff["s_0-s_1"] == pytest.approx(0.09, abs=0.01)


def test_run_metrics():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    results = run_metrics(predictions, test, [CV()], [TPR()])
    assert len(results) == 5
    assert results['TPR_s_0'] == approx(0.842857, RTOL)
    assert results['TPR_s_1'] == approx(0.886525, RTOL)
    assert results['TPR_s_0-s_1'] == approx(abs(0.842857 - 0.886525), RTOL)
    assert results['TPR_s_0/s_1'] == approx(0.842857 / 0.886525, RTOL)
    assert results['CV'] == approx(0.665)


def test_nmi_diff_non_binary_race():
    data: DataTuple = load_data(Adult("Race"))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    nmis = metric_per_sensitive_attribute(predictions, test, NMI())
    assert NMI().name == "NMI preds and y"
    test_dict = {'race_Amer-Indian-Eskimo_0': pytest.approx(0.14, abs=0.01),
                 'race_Amer-Indian-Eskimo_1': pytest.approx(0.41, abs=0.01),
                 'race_Asian-Pac-Islander_0': pytest.approx(0.14, abs=0.01),
                 'race_Asian-Pac-Islander_1': pytest.approx(0.10, abs=0.01),
                 'race_Black_0': pytest.approx(0.14, abs=0.01),
                 'race_Black_1': pytest.approx(0.15, abs=0.01),
                 'race_Other_0': pytest.approx(0.14, abs=0.01),
                 'race_Other_1': pytest.approx(0.18, abs=0.01),
                 'race_White_0': pytest.approx(0.16, abs=0.01),
                 'race_White_1': pytest.approx(0.14, abs=0.01)}

    for k, v in nmis.items():
        assert nmis[k] == test_dict[k]

    nmis_to_check = {k: nmis[k] for k in ('race_Amer-Indian-Eskimo_1',
                                          'race_Asian-Pac-Islander_1',
                                          'race_Black_1',
                                          'race_Other_1',
                                          'race_White_1')}
    nmi_diff = diff_per_sensitive_attribute(nmis_to_check)
    test_dict = {'race_Amer-Indian-Eskimo_1-race_Asian-Pac-Islander_1': pytest.approx(0.31, abs=0.01),
                 'race_Amer-Indian-Eskimo_1-race_Black_1': pytest.approx(0.26, abs=0.01),
                 'race_Amer-Indian-Eskimo_1-race_Other_1': pytest.approx(0.23, abs=0.01),
                 'race_Amer-Indian-Eskimo_1-race_White_1': pytest.approx(0.27, abs=0.01),
                 'race_Asian-Pac-Islander_1-race_Black_1': pytest.approx(0.04, abs=0.01),
                 'race_Asian-Pac-Islander_1-race_Other_1': pytest.approx(0.08, abs=0.01),
                 'race_Asian-Pac-Islander_1-race_White_1': pytest.approx(0.03, abs=0.01),
                 'race_Black_1-race_Other_1': pytest.approx(0.03, abs=0.01),
                 'race_Black_1-race_White_1': pytest.approx(0.02, abs=0.01),
                 'race_Other_1-race_White_1': pytest.approx(0.04, abs=0.01)}

    for k, v in nmi_diff.items():
        assert nmi_diff[k] == test_dict[k]


def test_tpr_diff_non_binary_race():
    data: DataTuple = load_data(Adult("Race"))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert TPR().name == "TPR"
    test_dict = {'race_Amer-Indian-Eskimo_0': pytest.approx(0.16, abs=0.01),
                 'race_Amer-Indian-Eskimo_1': pytest.approx(0.37, abs=0.01),
                 'race_Asian-Pac-Islander_0': pytest.approx(0.16, abs=0.01),
                 'race_Asian-Pac-Islander_1': pytest.approx(0.12, abs=0.01),
                 'race_Black_0': pytest.approx(0.16, abs=0.01),
                 'race_Black_1': pytest.approx(0.13, abs=0.01),
                 'race_Other_0': pytest.approx(0.16, abs=0.01),
                 'race_Other_1': pytest.approx(0.12, abs=0.01),
                 'race_White_0': pytest.approx(0.14, abs=0.01),
                 'race_White_1': pytest.approx(0.16, abs=0.01)}

    for k, v in tprs.items():
        assert tprs[k] == test_dict[k]

    tprs_to_check = {k: tprs[k] for k in ('race_Amer-Indian-Eskimo_1',
                                          'race_Asian-Pac-Islander_1',
                                          'race_Black_1',
                                          'race_Other_1',
                                          'race_White_1')}
    tpr_diff = diff_per_sensitive_attribute(tprs_to_check)
    test_dict = {'race_Amer-Indian-Eskimo_1-race_Asian-Pac-Islander_1': pytest.approx(0.25, abs=0.01),
                 'race_Amer-Indian-Eskimo_1-race_Black_1': pytest.approx(0.23, abs=0.01),
                 'race_Amer-Indian-Eskimo_1-race_Other_1': pytest.approx(0.25, abs=0.01),
                 'race_Amer-Indian-Eskimo_1-race_White_1': pytest.approx(0.20, abs=0.01),
                 'race_Asian-Pac-Islander_1-race_Black_1': pytest.approx(0.01, abs=0.01),
                 'race_Asian-Pac-Islander_1-race_Other_1': pytest.approx(0.00, abs=0.01),
                 'race_Asian-Pac-Islander_1-race_White_1': pytest.approx(0.04, abs=0.01),
                 'race_Black_1-race_Other_1': pytest.approx(0.01, abs=0.01),
                 'race_Black_1-race_White_1': pytest.approx(0.04, abs=0.01),
                 'race_Other_1-race_White_1': pytest.approx(0.04, abs=0.01)}

    for k, v in tpr_diff.items():
        assert tpr_diff[k] == test_dict[k]


def test_tpr_ratio_non_binary_race():
    data: DataTuple = load_data(Adult("Race"))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert TPR().name == "TPR"
    test_dict = {'race_Amer-Indian-Eskimo_0': pytest.approx(0.16, abs=0.01),
                 'race_Amer-Indian-Eskimo_1': pytest.approx(0.37, abs=0.01),
                 'race_Asian-Pac-Islander_0': pytest.approx(0.16, abs=0.01),
                 'race_Asian-Pac-Islander_1': pytest.approx(0.12, abs=0.01),
                 'race_Black_0': pytest.approx(0.16, abs=0.01),
                 'race_Black_1': pytest.approx(0.13, abs=0.01),
                 'race_Other_0': pytest.approx(0.16, abs=0.01),
                 'race_Other_1': pytest.approx(0.12, abs=0.01),
                 'race_White_0': pytest.approx(0.14, abs=0.01),
                 'race_White_1': pytest.approx(0.16, abs=0.01)}

    for k, v in tprs.items():
        assert tprs[k] == test_dict[k]

    tprs_to_check = {k: tprs[k] for k in ('race_Amer-Indian-Eskimo_1',
                                          'race_Asian-Pac-Islander_1',
                                          'race_Black_1',
                                          'race_Other_1',
                                          'race_White_1')}
    tpr_diff = ratio_per_sensitive_attribute(tprs_to_check)
    test_dict = {'race_Amer-Indian-Eskimo_1/race_Asian-Pac-Islander_1': pytest.approx(0.32, abs=0.1),
                 'race_Amer-Indian-Eskimo_1/race_Black_1': pytest.approx(0.37, abs=0.1),
                 'race_Amer-Indian-Eskimo_1/race_Other_1': pytest.approx(0.33, abs=0.1),
                 'race_Amer-Indian-Eskimo_1/race_White_1': pytest.approx(0.44, abs=0.1),
                 'race_Asian-Pac-Islander_1/race_Black_1': pytest.approx(0.88, abs=0.1),
                 'race_Asian-Pac-Islander_1/race_Other_1': pytest.approx(0.97, abs=0.1),
                 'race_Asian-Pac-Islander_1/race_White_1': pytest.approx(0.72, abs=0.1),
                 'race_Black_1/race_Other_1': pytest.approx(0.91, abs=0.1),
                 'race_Black_1/race_White_1': pytest.approx(0.74, abs=0.1),
                 'race_Other_1/race_White_1': pytest.approx(0.74, abs=0.1)}

    for k, v in tpr_diff.items():
        assert tpr_diff[k] == test_dict[k]


def test_nb_acc():
    data: DataTuple = load_data(NonBinaryToy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    acc_score = Accuracy().score(predictions, test)
    assert acc_score == 0.1
    accs = metric_per_sensitive_attribute(predictions, test, Accuracy())
    assert accs == {'sens_0': pytest.approx(0.09, abs=0.1), 'sens_1': pytest.approx(0.11, abs=0.1)}
    model = LR()
    predictions = model.run_test(train, test)
    acc_score = Accuracy().score(predictions, test)
    assert acc_score == 0.7
    accs = metric_per_sensitive_attribute(predictions, test, Accuracy())
    assert accs == {'sens_0': pytest.approx(0.72, abs=0.1), 'sens_1': pytest.approx(0.66, abs=0.1)}


def test_nb_tpr():
    data: DataTuple = load_data(NonBinaryToy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    tpr_score = TPR(pos_class=1).score(predictions, test)
    assert tpr_score == 0.0
    tpr_score = TPR(pos_class=2).score(predictions, test)
    assert tpr_score == 0.0
    tpr_score = TPR(pos_class=3).score(predictions, test)
    assert tpr_score == 1.0
    tpr_score = TPR(pos_class=4).score(predictions, test)
    assert tpr_score == 0.0
    tpr_score = TPR(pos_class=5).score(predictions, test)
    assert tpr_score == 0.0

    with pytest.raises(LabelOutOfBounds):
        tpr_score = TPR(pos_class=0).score(predictions, test)

    accs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert accs == {'sens_0': pytest.approx(0.0, abs=0.1), 'sens_1': pytest.approx(0.0, abs=0.1)}

    model = LR()
    predictions = model.run_test(train, test)

    print(list([(k, z) for k, z in zip(predictions.values, test.y.values) if k != z]))

    tpr_score = TPR(pos_class=1).score(predictions, test)
    assert tpr_score == 1.0
    tpr_score = TPR(pos_class=2).score(predictions, test)
    assert tpr_score == 0.0
    tpr_score = TPR(pos_class=3).score(predictions, test)
    assert tpr_score == 0.0
    tpr_score = TPR(pos_class=4).score(predictions, test)
    assert tpr_score == 1.0
    tpr_score = TPR(pos_class=5).score(predictions, test)
    assert tpr_score == 1.0

    with pytest.raises(LabelOutOfBounds):
        tpr_score = TPR(pos_class=0).score(predictions, test)

    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert tprs == {'sens_0': pytest.approx(1.0, abs=0.1), 'sens_1': pytest.approx(1.0, abs=0.1)}


def test_nb_tnr():
    data: DataTuple = load_data(NonBinaryToy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    tnr_score = TNR(pos_class=1).score(predictions, test)
    assert tnr_score == 1.0
    tnr_score = TNR(pos_class=2).score(predictions, test)
    assert tnr_score == 1.0
    tnr_score = TNR(pos_class=3).score(predictions, test)
    assert tnr_score == 0.0
    tnr_score = TNR(pos_class=4).score(predictions, test)
    assert tnr_score == 1.0
    tnr_score = TNR(pos_class=5).score(predictions, test)
    assert tnr_score == 1.0

    with pytest.raises(LabelOutOfBounds):
        tnr_score = TNR(pos_class=0).score(predictions, test)

    accs = metric_per_sensitive_attribute(predictions, test, TNR())
    assert accs == {'sens_0': pytest.approx(1.0, abs=0.1), 'sens_1': pytest.approx(1.0, abs=0.1)}

    model = LR()
    predictions = model.run_test(train, test)

    print(list([(k, z) for k, z in zip(predictions.values, test.y.values) if k != z]))

    tnr_score = TNR(pos_class=1).score(predictions, test)
    assert tnr_score == 1.0
    tnr_score = TNR(pos_class=2).score(predictions, test)
    assert tnr_score == 1.0
    tnr_score = TNR(pos_class=3).score(predictions, test)
    assert tnr_score == pytest.approx(0.7, abs=0.1)
    tnr_score = TNR(pos_class=4).score(predictions, test)
    assert tnr_score == pytest.approx(0.85, abs=0.1)
    tnr_score = TNR(pos_class=5).score(predictions, test)
    assert tnr_score == 1.0

    with pytest.raises(LabelOutOfBounds):
        tnr_score = TNR(pos_class=0).score(predictions, test)

    tnrs = metric_per_sensitive_attribute(predictions, test, TNR())
    assert tnrs == {'sens_0': pytest.approx(1.0, abs=0.1), 'sens_1': pytest.approx(1.0, abs=0.1)}
