"""
Runs given metrics on given algorithms for given datasets
"""

from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

from ethicml.algorithms.algorithm_base import Algorithm
from ethicml.data.dataset import Dataset
from ethicml.data.load import load_data
from ethicml.evaluators.per_sensitive_attribute import metric_per_sensitive_attribute
from ethicml.metrics.metric import Metric
from ethicml.preprocessing.train_test_split import train_test_split


def evaluate_models(datasets: List[Dataset], models: List[Algorithm],
                    metrics: List[Metric], per_sens_metrics: List[Metric]) -> pd.DataFrame:

    res = {}

    for dataset in datasets:
        data: Dict[str, pd.DataFrame] = load_data(dataset)

        train_test: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]] = train_test_split(data)
        train, test = train_test

        key_1 = dataset.get_dataset_name()
        res[key_1] = {}

        for model in models:

            predictions: np.array = model.run(train, test)

            key_2 = model.name
            res[key_1][key_2] = {}

            for metric in metrics:
                res[key_1][key_2][metric.get_name()] = metric.score(predictions, test)

            for metric in per_sens_metrics:
                res[key_1][key_2][metric.get_name()] = metric_per_sensitive_attribute(predictions, test, metric)

    return pd.DataFrame.from_dict(res)
