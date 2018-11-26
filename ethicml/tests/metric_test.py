"""
Test that we can get some metrics on predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from ethicml.algorithms.algorithm import Algorithm
from ethicml.algorithms.svm import SVM
from ethicml.data.adult import Adult
from ethicml.data.load import load_data
from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.metric import Metric
from ethicml.evaluators.per_sensitive_attribute import metric_per_sensitive_attribute
from ethicml.evaluators.per_sensitive_attribute import diff_per_sensitive_attribute
from ethicml.metrics.tpr import TPR
from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.tests.run_algorithm_test import get_train_test


def test_get_acc_of_predictions():
    train, test = get_train_test()
    model: Algorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    acc: Metric = Accuracy()
    assert acc.get_name() == "Accuracy"
    score = acc.score(predictions, test)
    assert score == 0.88


def test_accuracy_per_sens_attr():
    train, test = get_train_test()
    model: Algorithm = SVM()
    predictions: pd.DataFrame = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, Accuracy())
    assert acc_per_sens == {'s_0': 0.8780487804878049, 's_1': 0.882051282051282}


def test_accuracy_per_sens_attr_non_binary():
    data: Dict[str, pd.DataFrame] = load_data(Adult("Nationality"))
    train_test: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]] = train_test_split(data)
    train, test = train_test
    model: Algorithm = SVM()
    predictions: np.array = model.run_test(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, Accuracy())
    assert acc_per_sens == {'native-country_Cambodia_0': 0.8053251408090117,
                            'native-country_Cambodia_1': 0.75,
                            'native-country_Canada_0': 0.8057494866529774,
                            'native-country_Canada_1': 0.6551724137931034,
                            'native-country_China_0': 0.805441478439425,
                            'native-country_China_1': 0.7586206896551724,
                            'native-country_Columbia_0': 0.8050456363449903,
                            'native-country_Columbia_1': 0.9444444444444444,
                            'native-country_Cuba_0': 0.8052161412876065,
                            'native-country_Cuba_1': 0.8333333333333334,
                            'native-country_Dominican-Republic_0': 0.8049256028732683,
                            'native-country_Dominican-Republic_1': 0.9583333333333334,
                            'native-country_Ecuador_0': 0.8052480524805248,
                            'native-country_Ecuador_1': 0.8461538461538461,
                            'native-country_El-Salvador_0': 0.8047853768741015,
                            'native-country_El-Salvador_1': 0.967741935483871,
                            'native-country_England_0': 0.8053732567678425,
                            'native-country_England_1': 0.7647058823529411,
                            'native-country_France_0': 0.8056779747873322,
                            'native-country_France_1': 0.5,
                            'native-country_Germany_0': 0.8056584362139918,
                            'native-country_Germany_1': 0.7346938775510204,
                            'native-country_Greece_0': 0.8052027857435478,
                            'native-country_Greece_1': 1.0,
                            'native-country_Guatemala_0': 0.8050856146826617,
                            'native-country_Guatemala_1': 0.9375,
                            'native-country_Haiti_0': 0.8052081197457454,
                            'native-country_Haiti_1': 0.8666666666666667,
                            'native-country_Holand-Netherlands_0': 0.8053024874603337,
                            'native-country_Honduras_0': 0.8051828331455495,
                            'native-country_Honduras_1': 1.0,
                            'native-country_Hong_0': 0.8052227342549924,
                            'native-country_Hong_1': 1.0,
                            'native-country_Hungary_0': 0.8052626190232415,
                            'native-country_Hungary_1': 1.0,
                            'native-country_India_0': 0.8058521560574948,
                            'native-country_India_1': 0.6206896551724138,
                            'native-country_Iran_0': 0.8053478127241062,
                            'native-country_Iran_1': 0.75,
                            'native-country_Ireland_0': 0.8051828331455495,
                            'native-country_Ireland_1': 1.0,
                            'native-country_Italy_0': 0.8052961100277122,
                            'native-country_Italy_1': 0.8076923076923077,
                            'native-country_Jamaica_0': 0.8049856380796061,
                            'native-country_Jamaica_1': 0.9523809523809523,
                            'native-country_Japan_0': 0.8052134646962233,
                            'native-country_Japan_1': 0.84,
                            'native-country_Laos_0': 0.8052426786811386,
                            'native-country_Laos_1': 1.0,
                            'native-country_Mexico_0': 0.8024008350730689,
                            'native-country_Mexico_1': 0.9523809523809523,
                            'native-country_Nicaragua_0': 0.8051828331455495,
                            'native-country_Nicaragua_1': 1.0,
                            'native-country_Outlying-US(Guam-USVI-etc)_0': 0.8052426786811386,
                            'native-country_Outlying-US(Guam-USVI-etc)_1': 1.0,
                            'native-country_Peru_0': 0.8051429156848684,
                            'native-country_Peru_1': 1.0,
                            'native-country_Philippines_0': 0.8050113425448546,
                            'native-country_Philippines_1': 0.8450704225352113,
                            'native-country_Poland_0': 0.8053133654733818,
                            'native-country_Poland_1': 0.8,
                            'native-country_Portugal_0': 0.8052707136997539,
                            'native-country_Portugal_1': 0.8235294117647058,
                            'native-country_Puerto-Rico_0': 0.804830421377184,
                            'native-country_Puerto-Rico_1': 0.9230769230769231,
                            'native-country_Scotland_0': 0.8053052027857436,
                            'native-country_Scotland_1': 0.8,
                            'native-country_South_0': 0.8049856380796061,
                            'native-country_South_1': 0.9523809523809523,
                            'native-country_Taiwan_0': 0.8053904488624718,
                            'native-country_Taiwan_1': 0.7272727272727273,
                            'native-country_Thailand_0': 0.8053052027857436,
                            'native-country_Thailand_1': 0.8,
                            'native-country_Trinadad&Tobago_0': 0.8051828331455495,
                            'native-country_Trinadad&Tobago_1': 1.0,
                            'native-country_United-States_0': 0.8597914252607184,
                            'native-country_United-States_1': 0.8000224567707164,
                            'native-country_Vietnam_0': 0.8049506984387839,
                            'native-country_Vietnam_1': 0.9090909090909091,
                            'native-country_Yugoslavia_0': 0.8056750665847162,
                            'native-country_Yugoslavia_1': 0.2857142857142857}


def test_accuracy_per_sens_attr_non_binary_race():
    data: Dict[str, pd.DataFrame] = load_data(Adult("Race"))
    train_test: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]] = train_test_split(data)
    train, test = train_test
    model: Algorithm = SVM()
    predictions: np.array = model.run_test(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, Accuracy())
    assert acc_per_sens == {'race_Amer-Indian-Eskimo_0': 0.8169043190741889,
                            'race_Amer-Indian-Eskimo_1': 0.8681318681318682,
                            'race_Asian-Pac-Islander_0': 0.8182877728332274,
                            'race_Asian-Pac-Islander_1': 0.7915407854984894,
                            'race_Black_0': 0.8081395348837209,
                            'race_Black_1': 0.8986960882647944,
                            'race_Other_0': 0.8163897203013727,
                            'race_Other_1': 0.9375,
                            'race_White_0': 0.875250166777852,
                            'race_White_1': 0.8068923821039903}


def test_tpr_diff():
    train, test = get_train_test()
    model: Algorithm = SVM()
    predictions: np.array = model.run(train, test)
    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert TPR().get_name() == "TPR"
    assert tprs == {'s_0': 0.7704918032786885, 's_1': 0.9312977099236641}
    tpr_diff = diff_per_sensitive_attribute(tprs)
    assert tpr_diff["s_0-s_1"] == 0.16080590664497563


def test_tpr_diff_non_binary_race():
    data: Dict[str, pd.DataFrame] = load_data(Adult("Race"))
    train_test: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]] = train_test_split(data)
    train, test = train_test
    model: Algorithm = SVM()
    predictions: np.array = model.run_test(train, test)
    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert TPR().get_name() == "TPR"
    assert tprs == {'race_Amer-Indian-Eskimo_0': 0.3243589743589744,
                    'race_Amer-Indian-Eskimo_1': 0.5333333333333333,
                    'race_Asian-Pac-Islander_0': 0.32316534040671974,
                    'race_Asian-Pac-Islander_1': 0.3870967741935484,
                    'race_Black_0': 0.3268971710821733,
                    'race_Black_1': 0.3046875,
                    'race_Other_0': 0.325809199318569,
                    'race_Other_1': 0.2857142857142857,
                    'race_White_0': 0.3497942386831276,
                    'race_White_1': 0.3229166666666667}
    tprs_to_check = {k: tprs[k] for k in ('race_Amer-Indian-Eskimo_1',
                                          'race_Asian-Pac-Islander_1',
                                          'race_Black_1',
                                          'race_Other_1',
                                          'race_White_1')}
    tpr_diff = diff_per_sensitive_attribute(tprs_to_check)
    assert tpr_diff == {'race_Amer-Indian-Eskimo_1-race_Asian-Pac-Islander_1': 0.14623655913978495,
                        'race_Amer-Indian-Eskimo_1-race_Black_1': 0.22864583333333333,
                        'race_Amer-Indian-Eskimo_1-race_Other_1': 0.24761904761904763,
                        'race_Amer-Indian-Eskimo_1-race_White_1': 0.21041666666666664,
                        'race_Asian-Pac-Islander_1-race_Black_1': 0.08240927419354838,
                        'race_Asian-Pac-Islander_1-race_Other_1': 0.10138248847926268,
                        'race_Asian-Pac-Islander_1-race_White_1': 0.06418010752688169,
                        'race_Black_1-race_Other_1': 0.0189732142857143,
                        'race_Black_1-race_White_1': 0.018229166666666685,
                        'race_Other_1-race_White_1': 0.03720238095238099}
