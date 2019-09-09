"""
Runs given metrics on given algorithms for given datasets
"""
from pathlib import Path
from typing import List, Dict, Union, Sequence, Optional, Tuple
from collections import OrderedDict

import pandas as pd
from tqdm import tqdm

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithm
from ethicml.utility.data_structures import DataTuple, TestTuple, TrainTestPair, Results
from .parallelism import run_in_parallel
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
    poss_values: List[str] = []
    for col in train.s.columns:
        uniques = train.s[col].unique()
        for unique in uniques:
            poss_values.append(f"{col}_{unique}")

    return [f"{s}_{m.name}" for s in poss_values for m in metrics]


def per_sens_metrics_check(per_sens_metrics: Sequence[Metric]) -> None:
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
    metrics: Sequence[Metric] = (),
    per_sens_metrics: Sequence[Metric] = (),
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


def load_results(
    dataset_name: str,
    transform_name: str,
    topic: Optional[str] = None,
    outdir: Path = Path(".") / "results",
) -> Optional[Results]:
    """Load results from a CSV file that was created by `evaluate_models`

    Args:
        dataset_name: name of the dataset of the results
        transform_name: name of the transformation that was used for the results
        topic: (optional) topic string of the results
        outdir: directory where the results are stored

    Returns:
        DataFrame if the file exists; None otherwise
    """
    path_to_file = _result_path(outdir, dataset_name, transform_name, topic)
    return Results.from_file(path_to_file)


def _result_path(
    outdir: Path, dataset_name: str, transform_name: str, topic: Optional[str]
) -> Path:
    base_name: str = "" if topic is None else f"{topic}_"
    return outdir / f"{base_name}{dataset_name}_{transform_name}.csv"


def evaluate_models(
    datasets: List[Dataset],
    preprocess_models: Sequence[PreAlgorithm] = (),
    inprocess_models: Sequence[InAlgorithm] = (),
    postprocess_models: Sequence[PostAlgorithm] = (),
    metrics: Sequence[Metric] = (),
    per_sens_metrics: Sequence[Metric] = (),
    repeats: int = 1,
    test_mode: bool = False,
    delete_prev: bool = False,
    proportional_splits: bool = False,
    topic: Optional[str] = None,
    start_seed: int = 0,
) -> Results:
    """Evaluate all the given models for all the given datasets and compute all the given metrics

    Args:
        repeats: number of repeats to perform for the experiments
        datasets: list of dataset objects
        preprocess_models: list of preprocess model objects
        inprocess_models: list of inprocess model objects
        postprocess_models: list of postprocess model objects
        metrics: list of metric objects
        per_sens_metrics: list of metric objects that will be evaluated per sensitive attribute
        test_mode: if True, only use a small subset of the data so that the models run faster
        delete_prev:  False by default. If True, delete saved results in directory
        proportional_splits: if True, the train-test split preserves the proportion of s and y
        topic: (optional) a string that identifies the run; the string is prepended to the filename
        start_seed: random seed for the first repeat
    """
    # pylint: disable=too-many-arguments
    per_sens_metrics_check(per_sens_metrics)

    columns = ["dataset", "transform", "model", "repeat"]

    total_experiments = (
        len(datasets)
        * repeats
        * (len(preprocess_models) + ((1 + len(preprocess_models)) * len(inprocess_models)))
    )

    outdir = Path(".") / "results"  # OS-independent way of saying '../results'
    outdir.mkdir(exist_ok=True)

    if delete_prev:
        for dataset in datasets:
            transform_list = ["no_transform"]
            for preprocess_model in preprocess_models:
                transform_list.append(preprocess_model.name)
            for transform_name in transform_list:
                path_to_file: Path = _result_path(outdir, dataset.name, transform_name, topic)
                if path_to_file.exists():
                    path_to_file.unlink()

    pbar = tqdm(total=total_experiments)
    for dataset in datasets:
        # reset the seed for every dataset, otherwise seed would depend on the number of datasets
        seed = start_seed

        # ================================== begin: one repeat ====================================
        for repeat in range(repeats):
            train: DataTuple
            test: DataTuple
            train, test = train_test_split(
                load_data(dataset), 0.8, random_seed=seed, proportional=proportional_splits
            )
            if test_mode:
                # take smaller subset of training data to speed up training
                train = train.get_subset()

            to_operate_on: Dict[str, TrainTestPair] = {
                "no_transform": TrainTestPair(train=train, test=test)
            }

            # ========================== begin: run preprocessing models ==========================
            for pre_process_method in preprocess_models:
                logging: 'OrderedDict[str, str]' = OrderedDict()
                logging['model'] = pre_process_method.name
                logging['dataset'] = dataset.name
                logging['repeat'] = str(repeat)
                pbar.set_postfix(ordered_dict=logging)

                new_train, new_test = pre_process_method.run(train, test)
                to_operate_on[pre_process_method.name] = TrainTestPair(
                    train=new_train, test=new_test
                )

                pbar.update()
            # =========================== end: run preprocessing models ===========================

            # ========================= begin: loop over preprocessed data ========================
            for transform_name, transform in to_operate_on.items():

                transformed_train: DataTuple = transform.train
                transformed_test: Union[DataTuple, TestTuple] = transform.test
                results_df = pd.DataFrame(columns=columns)

                # ========================== begin: run inprocess models ==========================
                for model in inprocess_models:
                    logging = OrderedDict()
                    logging['model'] = model.name
                    logging['dataset'] = dataset.name
                    logging['transform'] = transform_name
                    logging['repeat'] = str(repeat)
                    pbar.set_postfix(ordered_dict=logging)

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

                    results_df = results_df.append(temp_res, ignore_index=True, sort=False)
                    pbar.update()
                # =========================== end: run inprocess models ===========================

                path_to_file = _result_path(outdir, dataset.name, transform_name, topic)
                results: Results = Results(results_df)
                # put old results before new results -> prepend=True
                results.append_from_file(path_to_file, prepend=True)
                results.save_as_csv(path_to_file)
            # ========================== end: loop over preprocessed data =========================
            seed += 2410
        # =================================== end: one repeat =====================================

    pbar.close()  # very important! when we're not using "with", we have to close tqdm manually

    preprocess_names = [model.name for model in preprocess_models]
    results = Results()  # create empty Results object
    for dataset in datasets:
        for transform_name in ["no_transform"] + preprocess_names:
            path_to_file = _result_path(outdir, dataset.name, transform_name, topic)
            results.append_from_file(path_to_file)
    return results


def evaluate_models_parallel(
    datasets: List[Dataset],
    inprocess_models: Sequence[InAlgorithm] = (),
    metrics: Sequence[Metric] = (),
    per_sens_metrics: Sequence[Metric] = (),
    repeats: int = 1,
    test_mode: bool = False,
    proportional_splits: bool = False,
    topic: Optional[str] = None,
    start_seed: int = 0,
    max_parallel: int = 0,
) -> Results:
    """Evaluate all the given models for all the given datasets and compute all the given metrics

    Args:
        datasets: list of dataset objects
        inprocess_models: list of inprocess model objects
        metrics: list of metric objects
        per_sens_metrics: list of metric objects that will be evaluated per sensitive attribute
        repeats: number of repeats to perform for the experiments
        test_mode: if True, only use a small subset of the data so that the models run faster
        proportional_splits: if True, the train-test split preserves the proportion of s and y
        topic: (optional) a string that identifies the run; the string is prepended to the filename
        start_seed: random seed for the first repeat
    """
    # pylint: disable=too-many-arguments
    per_sens_metrics_check(per_sens_metrics)

    transform_name = "no_transform"
    columns = ["dataset", "transform", "model", "repeat"]

    outdir = Path(".") / "results"  # OS-independent way of saying './results'
    outdir.mkdir(exist_ok=True)

    data_splits: List[TrainTestPair] = []
    test_data: List[Tuple[DataTuple, str, str]] = []
    for dataset in datasets:
        # reset the seed for every dataset, otherwise seed would depend on the number of datasets
        seed = start_seed

        for repeat in range(repeats):
            train: DataTuple
            test: DataTuple
            train, test = train_test_split(
                load_data(dataset), 0.8, random_seed=seed, proportional=proportional_splits
            )
            if test_mode:
                # take smaller subset of training data to speed up training
                train = train.get_subset()
            train = train.make_copy_with(name=f"{train.name} ({repeat})")
            data_splits.append(TrainTestPair(train, test))
            test_data.append((test, dataset.name, f"{repeat}-{seed}"))
            seed += 2410

    all_predictions = run_in_parallel(list(inprocess_models), data_splits, max_parallel)

    # transpose `all_results`
    all_predictions_t = [list(i) for i in zip(*all_predictions)]

    all_results = Results()

    # compute metrics, collect them and write them to files
    for preds_for_model, (test, dataset_name, repeat_str) in zip(all_predictions_t, test_data):
        results_df = pd.DataFrame(columns=columns)  # create empty results dataframe
        predictions: pd.DataFrame
        for predictions, model in zip(preds_for_model, inprocess_models):
            temp_res: Dict[str, Union[str, float]] = {
                "dataset": dataset_name,
                "transform": transform_name,
                "model": model.name,
                "repeat": repeat_str,
            }

            temp_res.update(run_metrics(predictions, test, metrics, per_sens_metrics))

            results_df = results_df.append(temp_res, ignore_index=True, sort=False)

        # write results to CSV files and load previous results from the files if they already exist
        path_to_file = _result_path(outdir, dataset_name, transform_name, topic)
        results: Results = Results(results_df)
        # put old results before new results -> prepend=True
        results.append_from_file(path_to_file, prepend=True)
        results.save_as_csv(path_to_file)
        all_results.append_df(results.data)

    return all_results
