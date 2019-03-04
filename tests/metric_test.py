"""
Test that we can get some metrics on predictions
"""

from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.logistic_regression import LRProb
from ethicml.algorithms.inprocess.svm import SVM
from ethicml.algorithms.utils import DataTuple
from ethicml.data import Adult
from ethicml.data.load import load_data
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.evaluators.per_sensitive_attribute import (
    metric_per_sensitive_attribute, diff_per_sensitive_attribute,
    ratio_per_sensitive_attribute, MetricNotApplicable
)
from ethicml.metrics import (Accuracy, BCR, CV, EqOppProbPos, Metric, NMI, PPV, NPV, ProbNeg,
                             ProbOutcome, ProbPos, TNR, TPR)
from ethicml.metrics.hsic import Hsic
from ethicml.preprocessing.train_test_split import train_test_split
from tests.run_algorithm_test import get_train_test


def test_get_acc_of_predictions():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    acc: Metric = Accuracy()
    assert acc.name == "Accuracy"
    score = acc.score(predictions, test)
    assert score == 0.89


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


def test_eqopp_per_sens_attr():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, EqOppProbPos())
    assert acc_per_sens == {'s_0': 0.8428571428571429, 's_1': 0.8865248226950354}


def test_proboutcome_per_sens_attr():
    train, test = get_train_test()
    model: InAlgorithm = LRProb()
    predictions: pd.DataFrame = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, ProbOutcome())
    assert acc_per_sens == {'s_0': 0.37192686620176246, 's_1': 0.661164329929575}


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
    assert acc_per_sens == {'native-country_Cambodia_0': 0.8019862803317293,
                            'native-country_Cambodia_1': 0.5,
                            'native-country_Canada_0': 0.8023028683047189,
                            'native-country_Canada_1': 0.7142857142857143,
                            'native-country_China_0': 0.8021763679293707,
                            'native-country_China_1': 0.7142857142857143,
                            'native-country_Columbia_0': 0.8017435897435897,
                            'native-country_Columbia_1': 0.8947368421052632,
                            'native-country_Cuba_0': 0.8018471010774756,
                            'native-country_Cuba_1': 0.8333333333333334,
                            'native-country_Dominican-Republic_0': 0.8016003282724662,
                            'native-country_Dominican-Republic_1': 0.9523809523809523,
                            'native-country_Ecuador_0': 0.8018848596599057,
                            'native-country_Ecuador_1': 0.8571428571428571,
                            'native-country_El-Salvador_0': 0.8016215106732348,
                            'native-country_El-Salvador_1': 0.92,
                            'native-country_England_0': 0.8022587268993839,
                            'native-country_England_1': 0.6896551724137931,
                            'native-country_France_0': 0.8019670115766827,
                            'native-country_France_1': 0.75,
                            'native-country_Germany_0': 0.8018916418217333,
                            'native-country_Germany_1': 0.8095238095238095,
                            'native-country_Greece_0': 0.8021730217302173,
                            'native-country_Greece_1': 0.6153846153846154,
                            'native-country_Guatemala_0': 0.801742696053306,
                            'native-country_Guatemala_1': 0.9285714285714286,
                            'native-country_Haiti_0': 0.8018461538461539,
                            'native-country_Haiti_1': 0.8421052631578947,
                            'native-country_Holand-Netherlands_0': 0.8019244549083837,
                            'native-country_Honduras_0': 0.8018230233510856,
                            'native-country_Honduras_1': 1.0,
                            'native-country_Hong_0': 0.8019872976849006,
                            'native-country_Hong_1': 0.7142857142857143,
                            'native-country_Hungary_0': 0.8018230233510856,
                            'native-country_Hungary_1': 1.0,
                            'native-country_India_0': 0.8019924001232412,
                            'native-country_India_1': 0.78125,
                            'native-country_Iran_0': 0.8018858255611356,
                            'native-country_Iran_1': 0.8333333333333334,
                            'native-country_Ireland_0': 0.8020075796374065,
                            'native-country_Ireland_1': 0.6666666666666666,
                            'native-country_Italy_0': 0.8018867924528302,
                            'native-country_Italy_1': 0.8235294117647058,
                            'native-country_Jamaica_0': 0.8019080837094789,
                            'native-country_Jamaica_1': 0.8095238095238095,
                            'native-country_Japan_0': 0.8020512820512821,
                            'native-country_Japan_1': 0.7368421052631579,
                            'native-country_Laos_0': 0.801863608437436,
                            'native-country_Laos_1': 1.0,
                            'native-country_Mexico_0': 0.7986605274173294,
                            'native-country_Mexico_1': 0.9483568075117371,
                            'native-country_Nicaragua_0': 0.8018645630570638,
                            'native-country_Nicaragua_1': 0.875,
                            'native-country_Outlying-US(Guam-USVI-etc)_0': 0.8018838947476196,
                            'native-country_Outlying-US(Guam-USVI-etc)_1': 1.0,
                            'native-country_Peru_0': 0.8019061283049805,
                            'native-country_Peru_1': 0.8181818181818182,
                            'native-country_Philippines_0': 0.8024716786817714,
                            'native-country_Philippines_1': 0.711864406779661,
                            'native-country_Poland_0': 0.8019080837094789,
                            'native-country_Poland_1': 0.8095238095238095,
                            'native-country_Portugal_0': 0.8019274143940948,
                            'native-country_Portugal_1': 0.8,
                            'native-country_Puerto-Rico_0': 0.8011719954765086,
                            'native-country_Puerto-Rico_1': 0.9761904761904762,
                            'native-country_Scotland_0': 0.8018433179723502,
                            'native-country_Scotland_1': 1.0,
                            'native-country_South_0': 0.8018258282900811,
                            'native-country_South_1': 0.85,
                            'native-country_Taiwan_0': 0.8018664752333095,
                            'native-country_Taiwan_1': 0.8333333333333334,
                            'native-country_Thailand_0': 0.8021516393442623,
                            'native-country_Thailand_1': 0.5555555555555556,
                            'native-country_Trinadad&Tobago_0': 0.8020278574354772,
                            'native-country_Trinadad&Tobago_1': 0.6,
                            'native-country_United-States_0': 0.8402298850574713,
                            'native-country_United-States_1': 0.7981795707382852,
                            'native-country_Vietnam_0': 0.8017232536670428,
                            'native-country_Vietnam_1': 0.9,
                            'native-country_Yugoslavia_0': 0.802068400573418,
                            'native-country_Yugoslavia_1': 0.3333333333333333}


def test_acc_per_race():
    data: DataTuple = load_data(Adult("Race"))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, Accuracy())
    assert acc_per_sens == {'race_Amer-Indian-Eskimo_0': 0.806688755435908,
                            'race_Amer-Indian-Eskimo_1': 0.8738738738738738,
                            'race_Asian-Pac-Islander_0': 0.8074928563869193,
                            'race_Asian-Pac-Islander_1': 0.80625,
                            'race_Black_0': 0.797752808988764,
                            'race_Black_1': 0.8966597077244259,
                            'race_Other_0': 00.8066281230642164,
                            'race_Other_1': 0.9036144578313253,
                            'race_White_0': 0.8756793478260869,
                            'race_White_1': 0.7953477160419429}


def test_tpr_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert TPR().name == "TPR"
    assert tprs == {'s_0': 0.8428571428571429, 's_1': 0.8865248226950354}
    tpr_diff = diff_per_sensitive_attribute(tprs)
    print(tpr_diff)
    assert tpr_diff["s_0-s_1"] == 0.04366767983789255


def test_get_nmi_of_predictions():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    nmi: Metric = NMI()
    assert nmi.name == "NMI"
    score = nmi.score(predictions, test)
    assert score == 0.5033012517022314


def test_nmi_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    nmis = metric_per_sensitive_attribute(predictions, test, NMI())
    assert NMI().name == "NMI"
    assert nmis == {'s_0': 0.5216526052290117, 's_1': 0.4172930822875038}
    nmi_diff = diff_per_sensitive_attribute(nmis)
    assert nmi_diff["s_0-s_1"] == 0.10435952294150791


def test_ppv_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    results = metric_per_sensitive_attribute(predictions, test, PPV())
    assert PPV().name == "PPV"
    assert results == {'s_0': 0.8805970149253731, 's_1': 0.9328358208955224}
    diff = diff_per_sensitive_attribute(results)
    assert diff["s_0-s_1"] == 0.052238805970149294


def test_npv_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    results = metric_per_sensitive_attribute(predictions, test, NPV())
    assert NPV().name == "NPV"
    assert results == {'s_0': 0.9172932330827067, 's_1': 0.7575757575757576}
    diff = diff_per_sensitive_attribute(results)
    assert diff["s_0-s_1"] == 0.15971747550694915


def test_bcr_diff():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    results = metric_per_sensitive_attribute(predictions, test, BCR())
    assert BCR().name == "BCR"
    assert results == {'s_0': 0.8906593406593406, 's_1': 0.8669912249068397}
    diff = diff_per_sensitive_attribute(results)
    assert diff["s_0-s_1"] == 0.02366811575250094


def test_cv():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    cv = CV()
    score = cv.score(predictions, test)
    assert CV().name == "CV"
    assert score == 0.665


def test_hsic():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    hsic = Hsic()
    score = hsic.score(predictions['preds'], test)
    assert Hsic().name == "HSIC"
    assert score == 0.023969453598182592


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
    assert NMI().name == "NMI"
    assert nmis == {'s_0': 0.9384615384615385, 's_1': 0.847457627118644}
    nmi_diff = diff_per_sensitive_attribute(nmis)
    assert nmi_diff["s_0-s_1"] == 0.09100391134289443


def test_run_metrics():
    train, test = get_train_test()
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    results = run_metrics(predictions, test, [CV()], [TPR()])
    np.testing.assert_allclose(results['s_0_TPR'], 0.8428571428571429)
    np.testing.assert_allclose(results['s_1_TPR'], 0.8865248226950354)
    np.testing.assert_allclose(results['CV'], 0.665)


def test_nmi_diff_non_binary_race():
    data: DataTuple = load_data(Adult("Race"))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    nmis = metric_per_sensitive_attribute(predictions, test, NMI())
    assert NMI().name == "NMI"
    assert nmis == {'race_Amer-Indian-Eskimo_0': 0.1438747853653249,
                    'race_Amer-Indian-Eskimo_1': 0.04870227433275973,
                    'race_Asian-Pac-Islander_0': 0.14246092044155623,
                    'race_Asian-Pac-Islander_1': 0.16851262040296772,
                    'race_Black_0': 0.13991195677147597,
                    'race_Black_1': 0.1597081623889905,
                    'race_Other_0': 0.14295105304509806,
                    'race_Other_1': 0.20971324033487476,
                    'race_White_0': 0.16907143394339794,
                    'race_White_1': 0.13865653944773082}
    nmis_to_check = {k: nmis[k] for k in ('race_Amer-Indian-Eskimo_1',
                                          'race_Asian-Pac-Islander_1',
                                          'race_Black_1',
                                          'race_Other_1',
                                          'race_White_1')}
    nmi_diff = diff_per_sensitive_attribute(nmis_to_check)
    assert nmi_diff == {'race_Amer-Indian-Eskimo_1-race_Asian-Pac-Islander_1': 0.11981034607020799,
                        'race_Amer-Indian-Eskimo_1-race_Black_1': 0.11100588805623077,
                        'race_Amer-Indian-Eskimo_1-race_Other_1': 0.16101096600211504,
                        'race_Amer-Indian-Eskimo_1-race_White_1': 0.0899542651149711,
                        'race_Asian-Pac-Islander_1-race_Black_1': 0.008804458013977223,
                        'race_Asian-Pac-Islander_1-race_Other_1': 0.041200619931907045,
                        'race_Asian-Pac-Islander_1-race_White_1': 0.029856080955236897,
                        'race_Black_1-race_Other_1': 0.05000507794588427,
                        'race_Black_1-race_White_1': 0.021051622941259673,
                        'race_Other_1-race_White_1': 0.07105670088714394}


def test_tpr_diff_non_binary_race():
    data: DataTuple = load_data(Adult("Race"))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert TPR().name == "TPR"
    assert tprs == {'race_Amer-Indian-Eskimo_0': 0.2997830802603037,
                    'race_Amer-Indian-Eskimo_1': 0.15384615384615385,
                    'race_Asian-Pac-Islander_0': 0.2956171735241503,
                    'race_Asian-Pac-Islander_1': 0.3902439024390244,
                    'race_Black_0': 0.30127041742286753,
                    'race_Black_1': 0.2543859649122807,
                    'race_Other_0': 0.2989601386481802,
                    'race_Other_1': 0.3,
                    'race_White_0': 0.3013698630136986,
                    'race_White_1': 0.29871367317770364}
    tprs_to_check = {k: tprs[k] for k in ('race_Amer-Indian-Eskimo_1',
                                          'race_Asian-Pac-Islander_1',
                                          'race_Black_1',
                                          'race_Other_1',
                                          'race_White_1')}
    tpr_diff = diff_per_sensitive_attribute(tprs_to_check)
    assert tpr_diff == {'race_Amer-Indian-Eskimo_1-race_Asian-Pac-Islander_1': 0.23639774859287055,
                        'race_Amer-Indian-Eskimo_1-race_Black_1': 0.10053981106612686,
                        'race_Amer-Indian-Eskimo_1-race_Other_1': 0.14615384615384613,
                        'race_Amer-Indian-Eskimo_1-race_White_1': 0.1448675193315498,
                        'race_Asian-Pac-Islander_1-race_Black_1': 0.1358579375267437,
                        'race_Asian-Pac-Islander_1-race_Other_1': 0.09024390243902441,
                        'race_Asian-Pac-Islander_1-race_White_1': 0.09153022926132076,
                        'race_Black_1-race_Other_1': 0.045614035087719274,
                        'race_Black_1-race_White_1': 0.04432770826542293,
                        'race_Other_1-race_White_1': 0.0012863268222963464}


def test_tpr_ratio_non_binary_race():
    data: DataTuple = load_data(Adult("Race"))
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = SVM()
    predictions: pd.DataFrame = model.run_test(train, test)
    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert TPR().name == "TPR"
    assert tprs == {'race_Amer-Indian-Eskimo_0': 0.2997830802603037,
                    'race_Amer-Indian-Eskimo_1': 0.15384615384615385,
                    'race_Asian-Pac-Islander_0': 0.2956171735241503,
                    'race_Asian-Pac-Islander_1': 0.3902439024390244,
                    'race_Black_0': 0.30127041742286753,
                    'race_Black_1': 0.2543859649122807,
                    'race_Other_0': 0.2989601386481802,
                    'race_Other_1': 0.3,
                    'race_White_0': 0.3013698630136986,
                    'race_White_1': 0.29871367317770364}
    tprs_to_check = {k: tprs[k] for k in ('race_Amer-Indian-Eskimo_1',
                                          'race_Asian-Pac-Islander_1',
                                          'race_Black_1',
                                          'race_Other_1',
                                          'race_White_1')}
    tpr_diff = ratio_per_sensitive_attribute(tprs_to_check)
    assert tpr_diff == {'race_Amer-Indian-Eskimo_1/race_Asian-Pac-Islander_1': 0.3942307692307692,
                        'race_Amer-Indian-Eskimo_1/race_Black_1': 0.6047745358090185,
                        'race_Amer-Indian-Eskimo_1/race_Other_1': 0.5128205128205129,
                        'race_Amer-Indian-Eskimo_1/race_White_1': 0.5150288308183045,
                        'race_Asian-Pac-Islander_1/race_Black_1': 1.534062237174096,
                        'race_Asian-Pac-Islander_1/race_Other_1': 1.3008130081300815,
                        'race_Asian-Pac-Islander_1/race_White_1': 1.306414595246431,
                        'race_Black_1/race_Other_1': 0.8479532163742691,
                        'race_Black_1/race_White_1': 0.8516046895548282,
                        'race_Other_1/race_White_1': 1.0043062200956938}
