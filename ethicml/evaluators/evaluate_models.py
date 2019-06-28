"""
Runs given metrics on given algorithms for given datasets
"""
import os
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from tqdm import tqdm

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithm
from ethicml.utility.data_structures import DataTuple, get_subset, TestTuple, TrainTestPair
from ..data.dataset import Dataset
from ..data.load import load_data
from .per_sensitive_attribute import (
    metric_per_sensitive_attribute,
    MetricNotApplicable,
    diff_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
)
from ..metrics.metric import Metric
from ..preprocessing.train_test_split import train_test_split


def get_sensitive_combinations(metrics: List[Metric], train: DataTuple) -> List[str]:
    """Get all possible combinations of sensitive attribute and metrics"""
    poss_values = []
    for col in train.s.columns:
        uniques = train.s[col].unique()
        for unique in uniques:
            poss_values.append(f"{col}_{unique}")

    return [f"{s}_{m.name}" for s in poss_values for m in metrics]


def per_sens_metrics_check(per_sens_metrics: List[Metric]):
    """Check if the given metrics allow application per sensitive attribute"""
    for metric in per_sens_metrics:
        if not metric.apply_per_sensitive:
            raise MetricNotApplicable(
                f"Metric {metric.name} is not applicable per sensitive "
                f"attribute, apply to whole dataset instead"
            )


def run_metrics(
    predictions: pd.DataFrame,
    actual: DataTuple,
    metrics: List[Metric],
    per_sens_metrics: List[Metric],
) -> Dict[str, float]:
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
            result[f"{metric.name}_{key}"] = value
    return result  # SUGGESTION: we could return a DataFrame here instead of a dictionary


def evaluate_models(
    datasets: List[Dataset],
    preprocess_models: List[PreAlgorithm],
    inprocess_models: List[InAlgorithm],
    postprocess_models: List[PostAlgorithm],
    metrics: List[Metric],
    per_sens_metrics: List[Metric],
    repeats: int = 3,
    test_mode: bool = False,
) -> pd.DataFrame:
    """Evaluate all the given models for all the given datasets and compute all the given metrics

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

    columns = ["dataset", "transform", "model", "repeat"]
    columns += [metric.name for metric in metrics]
    results = pd.DataFrame(columns=columns)

    total_experiments = (
        len(datasets)
        * repeats
        * (1 + len(preprocess_models) + ((1 + len(preprocess_models)) * len(inprocess_models)))
    )

    seed = 0
    with tqdm(total=total_experiments) as pbar:
        for dataset in datasets:
            for repeat in range(repeats):
                train: DataTuple
                test: DataTuple
                train, test = train_test_split(load_data(dataset), random_seed=seed)
                seed += 2410
                if test_mode:
                    # take smaller subset of training data to speed up training
                    train = get_subset(train)

                to_operate_on: Dict[str, TrainTestPair] = {
                    "no_transform": TrainTestPair(train=train, test=test)
                }

                for pre_process_method in preprocess_models:
                    new_train, new_test = pre_process_method.run(train, test)
                    to_operate_on[pre_process_method.name] = TrainTestPair(
                        train=new_train, test=new_test
                    )

                    pbar.update()

                transform_name: str
                for transform_name, transform in to_operate_on.items():

                    transformed_train: DataTuple = transform.train
                    transformed_test: Union[DataTuple, TestTuple] = transform.test

                    for model in inprocess_models:

                        temp_res: Dict[str, Union[str, float]] = {
                            "dataset": dataset.name,
                            "transform": transform_name,
                            "model": model.name,
                            "repeat": f"{repeat}-{seed}",
                        }

                        predictions: pd.DataFrame
                        predictions = model.run(transformed_train, transformed_test)

                        temp_res.update(run_metrics(predictions, test, metrics, per_sens_metrics))

                        for postprocess in postprocess_models:
                            # Post-processing has yet to be defined
                            # - leaving blank until we have an implementation to work with
                            pass

                        pbar.update()

                        results = results.append(temp_res, ignore_index=True)
                    outdir = Path("..") / "results"  # OS-independent way of saying '../results'
                    outdir.mkdir(exist_ok=True)
                    path_to_file = outdir / f"{dataset.name}_{transform_name}.csv"
                    exists = os.path.isfile(path_to_file)
                    if exists:
                        loaded_results = pd.read_csv(path_to_file)
                        results = pd.concat([loaded_results, results])
                    results.to_csv(path_to_file, index=False)

                    pbar.update()

    results = results.set_index(["dataset", "transform", "model", "repeat"])
    return results
