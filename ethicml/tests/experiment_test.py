from typing import Dict, Tuple
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from ethicml.algorithms.algorithm import Algorithm
from ethicml.algorithms.feldman import Feldman
from ethicml.algorithms.representation_hsic import RepresentationHSIC
from ethicml.algorithms.svm import SVM
from ethicml.data.adult import Adult
from ethicml.data.load import load_data
from ethicml.data.test import Test
from ethicml.evaluators.per_sensitive_attribute import metric_per_sensitive_attribute, diff_per_sensitive_attribute
from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.prob_pos import ProbPos
from ethicml.metrics.tpr import TPR
from ethicml.preprocessing.train_test_split import train_test_split


def test_fair_repr():
    data: Dict[str, pd.DataFrame] = load_data(Test())
    train_test: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]] = train_test_split(data, train_percentage=0.6)

    train, test = train_test

    train_s_as_y = {
        'x': train['x'].reset_index(drop=True),
        's': train['s'].replace(0, -1).reset_index(drop=True),
        'y': train['s'].replace(0, -1).reset_index(drop=True)
    }

    test_s_as_y = {
        'x': test['x'].reset_index(drop=True),
        's': test['s'].replace(0, -1).reset_index(drop=True),
        'y': test['s'].replace(0, -1).reset_index(drop=True)
    }
    train_s_as_y['y'].columns=['y']
    test_s_as_y['y'].columns=['y']

    model: Algorithm = RepresentationHSIC()

    train_reps, test_reps = model.run(train, test)

    new_train = {
        'x': train_reps,
        's': train['s'],
        'y': train['y']
    }

    train_pred_s = {
        'x': train_reps,
        's': train_s_as_y['s'],
        'y': train_s_as_y['y']
    }

    test_pred_s = {
        'x': test_reps,
        's': test_s_as_y['s'],
        'y': test_s_as_y['y']
    }

    new_test = {
        'x': test_reps,
        's': test['s'],
        'y': test['y']
    }

    svm = SVM()
    predictions = svm.run(train_pred_s, test_pred_s)

    acc_per_sens = metric_per_sensitive_attribute(predictions, test_pred_s, Accuracy())
    # tpr_per_sens = metric_per_sensitive_attribute(predictions, test_pred_s, TPR())
    pp_per_sens = metric_per_sensitive_attribute(predictions, test_pred_s, ProbPos())

    print("Acc per sens:", acc_per_sens)
    # print("TPR per sens:", tpr_per_sens)
    print("P(Y=1|S)", pp_per_sens)

    acc = Accuracy()
    print("Accuracy", acc.score(predictions, test_pred_s))
    probs = metric_per_sensitive_attribute(predictions, test_pred_s, Accuracy())
    print("Acc diff", diff_per_sensitive_attribute(probs))

    svm = SVM()
    predictions = svm.run(new_train, new_test)

    acc_per_sens = metric_per_sensitive_attribute(predictions, new_test, Accuracy())
    # tpr_per_sens = metric_per_sensitive_attribute(predictions, new_test, TPR())
    pp_per_sens = metric_per_sensitive_attribute(predictions, new_test, ProbPos())

    print("Acc per sens:", acc_per_sens)
    # print("TPR per sens:", tpr_per_sens)
    print("P(Y=1|S)", pp_per_sens)

    acc = Accuracy()
    print("Accuracy", acc.score(predictions, new_test))
    probs = metric_per_sensitive_attribute(predictions, new_test, ProbPos())
    print("P(Y=1)", diff_per_sensitive_attribute(probs))

    print(pd.concat([predictions, new_test['s']], axis=1).corr('pearson').values)

    asd = pd.crosstab(predictions.values.flatten(), test_pred_s['s'].values.flatten())
    print("fair preds and s:", asd)
    print(chi2_contingency(asd.values))



    return 1


def test_feldman():
    data: Dict[str, pd.DataFrame] = load_data(Test())
    train_test: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]] = train_test_split(data, train_percentage=0.6)

    train, test = train_test

    train_s_as_y = {
        'x': train['x'].reset_index(drop=True),
        's': train['s'].replace(0, -1).reset_index(drop=True),
        'y': train['s'].replace(0, -1).reset_index(drop=True)
    }

    test_s_as_y = {
        'x': test['x'].reset_index(drop=True),
        's': test['s'].replace(0, -1).reset_index(drop=True),
        'y': test['s'].replace(0, -1).reset_index(drop=True)
    }
    train_s_as_y['y'].columns = ['y']
    test_s_as_y['y'].columns = ['y']

    model: Algorithm = Feldman()

    train_reps, test_reps = model.run(train, test)

    new_train = {
        'x': train_reps,
        's': train['s'],
        'y': train['y']
    }

    train_pred_s = {
        'x': train_reps,
        's': train_s_as_y['s'],
        'y': train_s_as_y['y']
    }

    test_pred_s = {
        'x': test_reps,
        's': test_s_as_y['s'],
        'y': test_s_as_y['y']
    }

    new_test = {
        'x': test_reps,
        's': test['s'],
        'y': test['y']
    }

    svm = SVM()
    predictions = svm.run(train_pred_s, test_pred_s)

    acc_per_sens = metric_per_sensitive_attribute(predictions, test_pred_s, Accuracy())
    tpr_per_sens = metric_per_sensitive_attribute(predictions, test_pred_s, TPR())
    pp_per_sens = metric_per_sensitive_attribute(predictions, test_pred_s, ProbPos())

    print("Acc per sens:", acc_per_sens)
    print("TPR per sens:", tpr_per_sens)
    print("P(Y=1|S)", pp_per_sens)

    acc = Accuracy()
    print("Accuracy", acc.score(predictions, test_pred_s))
    probs = metric_per_sensitive_attribute(predictions, test_pred_s, Accuracy())
    print("Acc diff", diff_per_sensitive_attribute(probs))

    svm = SVM()
    predictions = svm.run(new_train, new_test)

    acc_per_sens = metric_per_sensitive_attribute(predictions, new_test, Accuracy())
    tpr_per_sens = metric_per_sensitive_attribute(predictions, new_test, TPR())
    pp_per_sens = metric_per_sensitive_attribute(predictions, new_test, ProbPos())

    print("Acc per sens:", acc_per_sens)
    print("TPR per sens:", tpr_per_sens)
    print("P(Y=1|S)", pp_per_sens)

    acc = Accuracy()
    print("Accuracy", acc.score(predictions, new_test))
    probs = metric_per_sensitive_attribute(predictions, new_test, ProbPos())
    print("P(Y=1)", diff_per_sensitive_attribute(probs))

    print(pd.concat([predictions, new_test['s']], axis=1).corr('pearson').values)

    asd = pd.crosstab(predictions.values.flatten(), test_pred_s['s'].values.flatten())
    print("fair preds and s:", asd)
    print(chi2_contingency(asd.values))

    return 1