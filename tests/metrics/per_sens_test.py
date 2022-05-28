"""Test the 'per sensitive attribute' evaluations."""
from typing import Dict, NamedTuple, Tuple

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from ethicml import DataTuple, KernelType, Prediction, train_test_split
from ethicml.data import Adult, Dataset, NonBinaryToy, Toy, load_data
from ethicml.metrics import (
    TPR,
    Accuracy,
    Metric,
    ProbNeg,
    ProbOutcome,
    ProbPos,
    Theil,
    metric_per_sensitive_attribute,
)
from ethicml.models import LR, SVM, InAlgorithm
from tests.conftest import get_id


class PerSensMetricTest(NamedTuple):
    """Define a test for a metric applied per sensitive group."""

    dataset: Dataset
    classifier: InAlgorithm
    metric: Metric
    expected_values: Dict[str, float]


def test_issue_431():
    """This issue highlighted that error would be raised due to not all values existing in subsets of the data."""
    x = pd.DataFrame(np.random.randn(100), columns=["x"])
    s = pd.Series(np.random.randn(100), name="s")
    y = pd.Series(np.random.randint(0, 5, 100), name="y")
    data = DataTuple.from_df(x=x, s=s, y=y)
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = LR()
    predictions: Prediction = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(
        predictions, test, TPR(pos_class=1, labels=list(range(y.nunique())))
    )
    print(acc_per_sens)


PER_SENS = [
    PerSensMetricTest(
        dataset=Toy(),
        classifier=SVM(),
        metric=Accuracy(),
        expected_values={"sensitive-attr_0": 0.921, "sensitive-attr_1": 0.929},
    ),
    PerSensMetricTest(
        dataset=Toy(),
        classifier=SVM(),
        metric=ProbPos(),
        expected_values={"sensitive-attr_0": 0.368, "sensitive-attr_1": 0.738},
    ),
    PerSensMetricTest(
        dataset=Toy(),
        classifier=LR(),
        metric=ProbOutcome(),
        expected_values={"sensitive-attr_0": 0.375, "sensitive-attr_1": 0.693},
    ),
    PerSensMetricTest(
        dataset=Toy(),
        classifier=SVM(),
        metric=ProbNeg(),
        expected_values={"sensitive-attr_0": 0.632, "sensitive-attr_1": 0.262},
    ),
    PerSensMetricTest(
        dataset=Toy(),
        classifier=SVM(),
        metric=Accuracy(),
        expected_values={"sensitive-attr_0": 0.921, "sensitive-attr_1": 0.928},
    ),
    PerSensMetricTest(
        dataset=Adult(split=Adult.Splits.NATIONALITY),
        classifier=SVM(kernel=KernelType.linear),
        metric=Accuracy(),
        expected_values={
            "native-country_1": 0.649,
            "native-country_2": 0.850,
            "native-country_3": 0.933,
            "native-country_4": 0.913,
            "native-country_5": 0.867,
            "native-country_6": 0.867,
            "native-country_7": 0.950,
            "native-country_8": 0.950,
            "native-country_9": 0.750,
            "native-country_10": 0.755,
            "native-country_11": 0.636,
            "native-country_12": 0.952,
            "native-country_13": 0.929,
            "native-country_15": 1.0,
            "native-country_16": 1.0,
            "native-country_17": 1.0,
            "native-country_18": 0.765,
            "native-country_19": 0.909,
            "native-country_20": 0.818,
            "native-country_21": 0.783,
            "native-country_22": 0.867,
            "native-country_23": 0.840,
            "native-country_24": 1.0,
            "native-country_25": 0.963,
            "native-country_26": 0.875,
            "native-country_27": 1.0,
            "native-country_28": 1.0,
            "native-country_29": 0.887,
            "native-country_30": 0.750,
            "native-country_31": 0.9,
            "native-country_32": 0.921,
            "native-country_33": 1.0,
            "native-country_34": 0.905,
            "native-country_35": 0.714,
            "native-country_36": 0.75,
            "native-country_37": 1.0,
            "native-country_38": 0.849,
            "native-country_39": 0.857,
            "native-country_40": 0.714,
        },
    ),
    PerSensMetricTest(
        dataset=Adult(split=Adult.Splits.RACE),
        classifier=SVM(kernel=KernelType.linear),
        metric=Accuracy(),
        expected_values={
            "race_0": 0.937,
            "race_1": 0.846,
            "race_2": 0.908,
            "race_3": 0.915,
            "race_4": 0.844,
        },
    ),
    PerSensMetricTest(
        dataset=Toy(),
        classifier=SVM(kernel=KernelType.linear),
        metric=Theil(),
        expected_values={"sensitive-attr_1": 0.024, "sensitive-attr_0": 0.045},
    ),
    PerSensMetricTest(
        dataset=NonBinaryToy(),
        classifier=SVM(kernel=KernelType.linear),
        metric=Accuracy(),
        expected_values={"sensitive-attr_1_1": 0.167, "sensitive-attr_1_0": 0.158},
    ),
    PerSensMetricTest(
        dataset=NonBinaryToy(),
        classifier=LR(),
        metric=Accuracy(),
        expected_values={"sensitive-attr_1_1": 0.143, "sensitive-attr_1_0": 0.158},
    ),
]


@pytest.mark.parametrize("dataset,classifier,metric,expected_values", PER_SENS, ids=get_id)
def test_metric_per_sens_attr(
    dataset: Dataset, classifier: InAlgorithm, metric: Metric, expected_values: Dict[str, float]
):
    """Test accuracy per sens attr."""
    data: DataTuple = load_data(dataset)
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    model: InAlgorithm = classifier
    predictions: Prediction = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, metric)
    try:
        for key, value in acc_per_sens.items():
            assert value == approx(expected_values[key], abs=0.001)
    except AssertionError:
        print({key: round(value, 3) for key, value in acc_per_sens.items()})
        raise AssertionError

    acc_per_sens = metric_per_sensitive_attribute(predictions, test, metric, use_sens_name=False)
    try:
        for key, value in expected_values.items():
            # Check that the sensitive attribute name is now just 'S'.
            assert acc_per_sens[f"S_{''.join(key.split('_')[-1:])}"] == approx(value, abs=0.001)
    except AssertionError:
        print({key: round(value, 3) for key, value in acc_per_sens.items()})
        raise AssertionError
