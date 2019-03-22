"""
Runs given metrics on given algorithms for given datasets
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple, Any, Union

import pandas as pd
import numpy as np

from ethicml.algorithms.algorithm_base import Algorithm, load_dataframe
from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithm
from ethicml.algorithms.utils import DataTuple, PathTuple, write_data_tuple, get_subset
from ..data.dataset import Dataset
from ..data.load import load_data
from .per_sensitive_attribute import (metric_per_sensitive_attribute, MetricNotApplicable,
                                      diff_per_sensitive_attribute, ratio_per_sensitive_attribute)
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
            raise MetricNotApplicable(f"Metric {metric.name} is not applicable per sensitive "
                                      f"attribute, apply to whole dataset instead")


def run_metrics(predictions: pd.DataFrame, actual: DataTuple, metrics: List[Metric],
                per_sens_metrics: List[Metric]) -> Dict[str, float]:
    """Run all the given metrics on the given predictions and return the results

    Args:
        predictions: DataFrame with predictions
        actual: DataTuple with the labels
        metrics: list of metrics
        per_sens_metrics: list of metrics that are computed per sensitive attribute
    """
    result: Dict[str, float] = {}
    for metric in metrics:
        result[metric.name] = metric.score(predictions, actual)

    for metric in per_sens_metrics:
        per_sens = metric_per_sensitive_attribute(predictions, actual, metric)
        diff_per_sens = diff_per_sensitive_attribute(per_sens)
        ratio_per_sens = ratio_per_sensitive_attribute(per_sens)
        per_sens.update(diff_per_sens)
        per_sens.update(ratio_per_sens)
        for key, value in per_sens.items():
            result[f'{metric.name}_{key}'] = value
    return result  # SUGGESTION: we could return a DataFrame here instead of a dictionary


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
                'train': DataTuple(x=new_train, s=train.s, y=train.y),
                'test': DataTuple(x=new_test, s=test.s, y=test.y),
            }

        columns = ['model']
        columns += [metric.name for metric in metrics]
        columns += get_sensitive_combinations(per_sens_metrics, train)

        for name, transform in to_operate_on.items():
            results = pd.DataFrame(columns=columns)

            transformed_train: DataTuple = transform['train']
            transformed_test: DataTuple = transform['test']
            transform_name: str = name

            for model in inprocess_models:

                temp_res: Dict[str, Union[str, float]] = {'model': model.name}

                predictions: np.array
                if test_mode:
                    predictions = model.run_test(transformed_train, transformed_test)
                else:
                    predictions = model.run(transformed_train, transformed_test)

                temp_res.update(run_metrics(predictions, test, metrics, per_sens_metrics))

                for postprocess in postprocess_models:
                    # Post-processing has yet to be defined
                    # - leaving blank until we have an implementation to work with
                    pass

                results = results.append(temp_res, ignore_index=True)
            outdir = '../results'
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            results.to_csv(f"../results/{dataset.name}_{transform_name}.csv", index=False)


def evaluate_threaded_models(datasets: List[Dataset], preprocess_models: List[PreAlgorithm],
                             inprocess_models: List[InAlgorithm],
                             postprocess_models: List[PostAlgorithm], metrics: List[Metric],
                             per_sens_metrics: List[Metric], test_mode: bool = False):
    """Evaluate all the given models for all the given datasets and compute all the given metrics

    This function only works with threaded models.

    Args:
        datasets: list of dataset objects
        preprocess_models: list of preprocess model objects
        inprocess_models: list of inprocess model objects
        postprocess_models: list of postprocess model objects
        metrics: list of metric objects
        per_sens_metrics: list of metric objects that will be evaluated per sensitive attribute
        test_mode: if True, only use a small subset of the data so that the models run faster
    """
    per_sens_metrics_check(per_sens_metrics)

    for dataset in datasets:
        train: DataTuple
        test: DataTuple
        train, test = train_test_split(load_data(dataset))
        if test_mode:
            train = get_subset(train)  # take smaller subset of training data to speed up evaluation

        with TemporaryDirectory() as tmpdir:  # a new temp dir per dataset
            tmp_path = Path(tmpdir)

            # write the files for the non-transformed data
            # we apply a scope to the paths in order to avoid name duplications (not really needed)
            no_transform_dir = tmp_path / "no_transform"
            train_paths, test_paths = write_data_tuple(train, test, no_transform_dir)
            to_operate_on: Dict[str, Any] = {"no_transform": {
                'train_paths': train_paths, 'test_paths': test_paths, 'tmp_path': no_transform_dir}}

            for pre_process_method in preprocess_models:
                # separate directory for each preprocessing model
                model_dir = subdir_for_model(tmp_path, pre_process_method)
                new_train, new_test = pre_process_method.run_thread(train_paths, test_paths,
                                                                    model_dir)

                # construct a new path tuple with the new features
                new_train_paths = PathTuple(x=new_train, s=train_paths.s, y=train_paths.y)
                new_test_paths = PathTuple(x=new_test, s=test_paths.s, y=test_paths.y)

                to_operate_on[pre_process_method.name] = {'train_paths': new_train_paths,
                                                          'test_paths': new_test_paths,
                                                          'tmp_path': model_dir}

            columns = ['model']
            columns += [metric.name for metric in metrics]
            columns += get_sensitive_combinations(per_sens_metrics, train)

            transform_name: str
            for transform_name, transform in to_operate_on.items():
                results = pd.DataFrame(columns=columns)

                transformed_train: PathTuple = transform['train_paths']
                transformed_test: PathTuple = transform['test_paths']
                # separate directory for each transformed data set (this should already exist)
                transform_dir: Path = transform['tmp_path']

                for model in inprocess_models:
                    # separate directory for each model
                    model_dir = subdir_for_model(transform_dir, model)

                    pred_path: Path = model.run_thread(transformed_train, transformed_test,
                                                       model_dir)
                    predictions = load_dataframe(pred_path)
                    temp_res: Dict[str, Union[str, float]] = {'model': model.name}
                    temp_res.update(run_metrics(predictions, test, metrics, per_sens_metrics))

                    for postprocess in postprocess_models:
                        # Post-processing has yet to be defined
                        # - leaving blank until we have an implementation to work with
                        pass

                    results = results.append(temp_res, ignore_index=True)
                outdir = Path('..') / 'results'  # OS-independent way of saying '../results'
                outdir.mkdir(exist_ok=True)
                results.to_csv(outdir / f"{dataset.name}_{transform_name}.csv", index=False)


def subdir_for_model(parent_dir: Path, model: Algorithm) -> Path:
    """Create a subdirectory in `parent_dir` based on the name of the given model"""
    model_dir = parent_dir / model.name.strip().replace(' ', '_')
    model_dir.mkdir(parents=False, exist_ok=False)
    return model_dir
