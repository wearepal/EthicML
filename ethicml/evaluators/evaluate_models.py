"""
Runs given metrics on given algorithms for given datasets
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple, Any

import pandas as pd
import numpy as np

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.threaded.threaded_in_algorithm import ThreadedInAlgorithm
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithm
from ethicml.algorithms.utils import make_data_tuple, DataTuple, PathTuple
from ..data.dataset import Dataset
from ..data.load import load_data
from .per_sensitive_attribute import metric_per_sensitive_attribute, MetricNotApplicable
from ..metrics.metric import Metric
from ..preprocessing.train_test_split import train_test_split


def get_sensitive_combinations(metrics: List[Metric], train: DataTuple) -> List[str]:
    """

    Args:
        metrics:
        train:

    Returns:

    """
    poss_values = []
    for col in train.s:
        uniques = train.s[col].unique()
        for unique in uniques:
            poss_values.append(f"{col}_{unique}")

    return [f'{s}_{m.name}' for s in poss_values for m in metrics]


def per_sens_metrics_check(per_sens_metrics: List[Metric]):
    """

    Args:
        per_sens_metrics:
    """
    for metric in per_sens_metrics:
        if not metric.apply_per_sensitive:
            raise MetricNotApplicable()


def evaluate_models(datasets: List[Dataset], preprocess_models: List[PreAlgorithm],
                    inprocess_models: List[InAlgorithm], postprocess_models: List[PostAlgorithm],
                    metrics: List[Metric], per_sens_metrics: List[Metric],
                    test_mode: bool = False) -> None:
    """

    Args:
        datasets:
        preprocess_models:
        inprocess_models:
        postprocess_models:
        metrics:
        per_sens_metrics:
        test_mode:
    """
    per_sens_metrics_check(per_sens_metrics)

    for dataset in datasets:

        data: DataTuple = load_data(dataset)

        train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
        train, test = train_test

        to_operate_on = {"no_transform": {'train': train,
                                          'test': test}}

        for pre_process_method in preprocess_models:
            if test_mode:
                new_train, new_test = pre_process_method.run_test(train, test)
            else:
                new_train, new_test = pre_process_method.run(train, test)
            to_operate_on[pre_process_method.name] = {
                'train': make_data_tuple(new_train, train.s, train.y),
                'test': make_data_tuple(new_test, test.s, test.y),
            }

        columns = ['model']
        columns += [metric.name for metric in metrics]
        columns += get_sensitive_combinations(per_sens_metrics, train)

        temp_res: Dict[str, Any] = {}

        for name, transform in to_operate_on.items():
            results = pd.DataFrame(columns=columns)

            transformed_train: DataTuple = transform['train']
            transformed_test: DataTuple = transform['test']
            transform_name: str = name

            for model in inprocess_models:

                temp_res['model'] = model.name

                predictions: np.array
                if test_mode:
                    predictions = model.run_test(transformed_train, transformed_test)
                else:
                    predictions = model.run(transformed_train, transformed_test)

                for metric in metrics:
                    temp_res[metric.name] = metric.score(predictions, test)

                for metric in per_sens_metrics:
                    per_sens = metric_per_sensitive_attribute(predictions, test, metric)
                    for key, value in per_sens.items():
                        temp_res[f'{key}_{metric.name}'] = value

                for postprocess in postprocess_models:
                    # Post-processing has yet to be defined
                    # - leaving blank until we have an implementation to work with
                    pass

                results = results.append(temp_res, ignore_index=True)
            outdir = '../results'
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            results.to_csv(f"../results/{dataset.name}_{transform_name}.csv", index=False)


def call_on_saved_data(algorithm: ThreadedInAlgorithm, train: DataTuple, test: DataTuple) -> (
        pd.DataFrame):
    """
    This function saves the given dataframes to the harddrive (in separate files) and then calls the
    run function of the given algorithm with two path tuples of the paths to the saved files and a
    temporary directory.

    Args:
        algorithm: algorithm that will be called with the paths to the data
        data: Tuple of Pandas dataframe to be saved and passed as a tuple of file paths to
              the function
    Returns:
        the predictions of the algorithm
    """
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        train_paths: Dict[str, Path] = {}
        test_paths: Dict[str, Path] = {}
        for data_tuple, data_paths, prefix in [(train, train_paths, "train"),
                                               (test, test_paths, "test")]:
            data_dict = data_tuple._asdict()
            for key, data in data_dict.items():
                data_path = tmp_path / Path(f"data_{prefix}_{key}.parquet")
                data.to_parquet(data_path, compression=None)
                data_paths[key] = data_path
        # call the algorithm with the paths to the saved files and with the path to the temp dir
        result = algorithm.run(PathTuple(**train_paths), PathTuple(**test_paths), tmp_path)
    return result
