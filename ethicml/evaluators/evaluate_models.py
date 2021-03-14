"""Runs given metrics on given algorithms for given datasets."""
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import pandas as pd
from tqdm import tqdm

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithm
from ethicml.data.dataset import Dataset
from ethicml.data.load import load_data
from ethicml.metrics.metric import Metric
from ethicml.preprocessing import DataSplitter, RandomSplit, scale_continuous
from ethicml.utility import (
    DataTuple,
    Prediction,
    Results,
    ResultsAggregator,
    TestTuple,
    TrainTestPair,
    make_results,
)

from .parallelism import run_in_parallel
from .per_sensitive_attribute import (
    MetricNotApplicable,
    diff_per_sensitive_attribute,
    metric_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
)

__all__ = ["evaluate_models", "run_metrics", "load_results", "evaluate_models_async"]

from ..preprocessing.scaling import ScalerType


def get_sensitive_combinations(metrics: List[Metric], train: DataTuple) -> List[str]:
    """Get all possible combinations of sensitive attribute and metrics."""
    poss_values: List[str] = []
    for col in train.s.columns:
        uniques = train.s[col].unique()
        for unique in uniques:
            poss_values.append(f"{col}_{unique}")

    return [f"{s}_{m.name}" for s in poss_values for m in metrics]


def per_sens_metrics_check(per_sens_metrics: Sequence[Metric]) -> None:
    """Check if the given metrics allow application per sensitive attribute."""
    for metric in per_sens_metrics:
        if not metric.apply_per_sensitive:
            raise MetricNotApplicable(
                f"Metric {metric.name} is not applicable per sensitive "
                f"attribute, apply to whole dataset instead"
            )


def run_metrics(
    predictions: Prediction,
    actual: DataTuple,
    metrics: Sequence[Metric] = (),
    per_sens_metrics: Sequence[Metric] = (),
    diffs_and_ratios: bool = True,
) -> Dict[str, float]:
    """Run all the given metrics on the given predictions and return the results.

    Args:
        predictions: DataFrame with predictions
        actual: DataTuple with the labels
        metrics: list of metrics
        per_sens_metrics: list of metrics that are computed per sensitive attribute
        diffs_and_ratios: if True, compute diffs and ratios per sensitive attribute
    """
    result: Dict[str, float] = {}
    if predictions.hard.isna().any(axis=None):
        return {"algorithm_failed": 1.0}
    for metric in metrics:
        result[metric.name] = metric.score(predictions, actual)

    for metric in per_sens_metrics:
        per_sens = metric_per_sensitive_attribute(predictions, actual, metric)
        if diffs_and_ratios:
            diff_per_sens = diff_per_sensitive_attribute(per_sens)
            ratio_per_sens = ratio_per_sensitive_attribute(per_sens)
            per_sens.update(diff_per_sens)
            per_sens.update(ratio_per_sens)
        for key, value in per_sens.items():
            result[f"{metric.name}_{key}"] = value
    for key, value in predictions.info.items():
        result[key] = value
    return result  # SUGGESTION: we could return a DataFrame here instead of a dictionary


def load_results(
    dataset_name: str,
    transform_name: str,
    topic: Optional[str] = None,
    outdir: Path = Path(".") / "results",
) -> Optional[Results]:
    """Load results from a CSV file that was created by `evaluate_models`.

    Args:
        dataset_name: name of the dataset of the results
        transform_name: name of the transformation that was used for the results
        topic: (optional) topic string of the results
        outdir: directory where the results are stored

    Returns:
        DataFrame if the file exists; None otherwise
    """
    csv_file = _result_path(outdir, dataset_name, transform_name, topic)
    if csv_file.is_file():
        return make_results(csv_file)
    return None


def _result_path(
    outdir: Path, dataset_name: str, transform_name: str, topic: Optional[str]
) -> Path:
    base_name: str = "" if topic is None else f"{topic}_"
    return outdir / f"{base_name}{dataset_name}_{transform_name}.csv"


def _delete_previous_results(
    outdir: Path, datasets: List[Dataset], transforms: Sequence[PreAlgorithm], topic: Optional[str]
) -> None:
    for dataset in datasets:
        transform_list = ["no_transform"]
        for preprocess_model in transforms:
            transform_list.append(preprocess_model.name)
        for transform_name in transform_list:
            path_to_file: Path = _result_path(outdir, dataset.name, transform_name, topic)
            if path_to_file.exists():
                path_to_file.unlink()


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
    splitter: Optional[DataSplitter] = None,
    topic: Optional[str] = None,
    fair_pipeline: bool = True,
    scaler: Optional[ScalerType] = None,
) -> Results:
    """Evaluate all the given models for all the given datasets and compute all the given metrics.

    Args:
        datasets: list of dataset objects
        scaler: scaler to use on the continuous features of the dataset.
        preprocess_models: list of preprocess model objects
        inprocess_models: list of inprocess model objects
        postprocess_models: list of postprocess model objects
        metrics: list of metric objects
        per_sens_metrics: list of metric objects that will be evaluated per sensitive attribute
        repeats: number of repeats to perform for the experiments
        test_mode: if True, only use a small subset of the data so that the models run faster
        delete_prev:  False by default. If True, delete saved results in directory
        splitter: (optional) custom train-test splitter
        topic: (optional) a string that identifies the run; the string is prepended to the filename
        fair_pipeline: if True, run fair inprocess algorithms on the output of preprocessing
    """
    # pylint: disable=too-many-arguments
    per_sens_metrics_check(per_sens_metrics)
    train_test_split: DataSplitter
    if splitter is None:
        train_test_split = RandomSplit(train_percentage=0.8, start_seed=0)
    else:
        train_test_split = splitter

    columns = ["dataset", "scaler", "transform", "model", "split_id"]

    total_experiments = (
        len(datasets)
        * repeats
        * (len(preprocess_models) + ((1 + len(preprocess_models)) * len(inprocess_models)))
    )

    outdir = Path(".") / "results"
    outdir.mkdir(exist_ok=True)

    if delete_prev:
        _delete_previous_results(outdir, datasets, preprocess_models, topic)

    pbar = tqdm(total=total_experiments, smoothing=0)
    for dataset in datasets:
        # ================================== begin: one repeat ====================================
        for split_id in range(repeats):
            train: DataTuple
            test: DataTuple
            train, test, split_info = train_test_split(load_data(dataset), split_id=split_id)
            if scaler is not None:
                train, scaler_post = scale_continuous(dataset, train, scaler)
                test, _ = scale_continuous(dataset, test, scaler_post, fit=False)
            if test_mode:
                # take smaller subset of training data to speed up training
                train = train.get_subset()

            to_operate_on: Dict[str, TrainTestPair] = {
                "no_transform": TrainTestPair(train=train, test=test)
            }

            # ========================== begin: run preprocessing models ==========================
            for pre_process_method in preprocess_models:
                logging: "OrderedDict[str, str]" = OrderedDict()
                logging["model"] = pre_process_method.name
                logging["dataset"] = dataset.name
                logging["repeat"] = str(split_id)
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
                    if (
                        not fair_pipeline
                        and transform_name != "no_transform"
                        and model.is_fairness_algo
                    ):
                        pbar.update()
                        continue

                    logging = OrderedDict()
                    logging["model"] = model.name
                    logging["dataset"] = dataset.name
                    logging["transform"] = transform_name
                    logging["repeat"] = str(split_id)
                    pbar.set_postfix(ordered_dict=logging)

                    temp_res: Dict[str, Union[str, float]] = {
                        "dataset": dataset.name,
                        "scaler": "None" if scaler is None else scaler.__class__.__name__,
                        "transform": transform_name,
                        "model": model.name,
                        "split_id": split_id,
                        **split_info,
                    }

                    predictions: Prediction = model.run(transformed_train, transformed_test)

                    temp_res.update(run_metrics(predictions, test, metrics, per_sens_metrics))

                    for postprocess in postprocess_models:
                        # Post-processing has yet to be defined
                        # - leaving blank until we have an implementation to work with
                        pass

                    results_df = results_df.append(temp_res, ignore_index=True, sort=False)
                    pbar.update()
                # =========================== end: run inprocess models ===========================

                csv_file = _result_path(outdir, dataset.name, transform_name, topic)
                aggregator = ResultsAggregator(results_df)
                # put old results before new results -> prepend=True
                aggregator.append_from_csv(csv_file, prepend=True)
                aggregator.save_as_csv(csv_file)
            # ========================== end: loop over preprocessed data =========================
        # =================================== end: one repeat =====================================

    pbar.close()  # very important! when we're not using "with", we have to close tqdm manually

    preprocess_names = [model.name for model in preprocess_models]
    aggregator = ResultsAggregator()  # create empty aggregator object
    for dataset in datasets:
        for transform_name in ["no_transform"] + preprocess_names:
            csv_file = _result_path(outdir, dataset.name, transform_name, topic)
            aggregator.append_from_csv(csv_file)
    return aggregator.results


class _DataInfo(NamedTuple):
    test: DataTuple
    dataset_name: str
    transform_name: str
    split_info: Dict[str, float]
    scaler: str


async def evaluate_models_async(
    datasets: List[Dataset],
    preprocess_models: Sequence[PreAlgorithm] = (),
    inprocess_models: Sequence[InAlgorithm] = (),
    postprocess_models: Sequence[PostAlgorithm] = (),
    metrics: Sequence[Metric] = (),
    per_sens_metrics: Sequence[Metric] = (),
    repeats: int = 1,
    test_mode: bool = False,
    delete_prev: bool = False,
    splitter: Optional[DataSplitter] = None,
    topic: Optional[str] = None,
    fair_pipeline: bool = True,
    max_parallel: int = 1,
    scaler: Optional[ScalerType] = None,
) -> Results:
    """Evaluate all the given models for all the given datasets and compute all the given metrics.

    Args:
        datasets: list of dataset objects
        scaler: Sklearn-style scaler to be used on the continuous features.
        preprocess_models: list of preprocess model objects
        inprocess_models: list of inprocess model objects
        postprocess_models: list of postprocess model objects
        metrics: list of metric objects
        per_sens_metrics: list of metric objects that will be evaluated per sensitive attribute
        repeats: number of repeats to perform for the experiments
        test_mode: if True, only use a small subset of the data so that the models run faster
        delete_prev:  False by default. If True, delete saved results in directory
        splitter: (optional) custom train-test splitter
        topic: (optional) a string that identifies the run; the string is prepended to the filename
        fair_pipeline: if True, run fair inprocess algorithms on the output of preprocessing
        max_parallel: max number of threads ot run in parallel (default: 1)
    """
    # pylint: disable=too-many-arguments
    del postprocess_models  # not used at the moment
    per_sens_metrics_check(per_sens_metrics)
    if splitter is None:
        train_test_split: DataSplitter = RandomSplit(train_percentage=0.8, start_seed=0)
    else:
        train_test_split = splitter

    default_transform_name = "no_transform"

    outdir = Path(".") / "results"  # OS-independent way of saying './results'
    outdir.mkdir(exist_ok=True)

    if delete_prev:
        _delete_previous_results(outdir, datasets, preprocess_models, topic)

    all_results = ResultsAggregator()

    # ======================================= prepare data ========================================
    data_splits: List[TrainTestPair] = []
    test_data: List[_DataInfo] = []  # contains the test set and other things needed for the metrics
    for dataset in datasets:
        for split_id in range(repeats):
            train: DataTuple
            test: DataTuple
            train, test, split_info = train_test_split(load_data(dataset), split_id=split_id)
            if test_mode:
                # take smaller subset of training data to speed up training
                train = train.get_subset()
            train = train.replace(name=f"{train.name} ({split_id})")
            data_splits.append(TrainTestPair(train, test))
            split_info.update({"split_id": split_id})
            test_data.append(
                _DataInfo(
                    test=test,
                    dataset_name=dataset.name,
                    transform_name=default_transform_name,
                    split_info=split_info,
                    scaler="None" if scaler is None else scaler.__class__.__name__,
                )
            )

    # ============================= inprocess models on untransformed =============================
    all_predictions = await run_in_parallel(inprocess_models, data_splits, max_parallel)
    inprocess_untransformed = _gather_metrics(
        all_predictions, test_data, inprocess_models, metrics, per_sens_metrics, outdir, topic
    )
    all_results.append_df(inprocess_untransformed)

    # ===================================== preprocess models =====================================
    # run all preprocess models
    all_transformed = await run_in_parallel(preprocess_models, data_splits, max_parallel)

    # append the transformed data to `transformed_data`
    transformed_data: List[TrainTestPair] = []
    transformed_test: List[_DataInfo] = []
    for transformed, pre_model in zip(all_transformed, preprocess_models):
        for (transf_train, transf_test), data_info in zip(transformed, test_data):
            transformed_data.append(TrainTestPair(transf_train, transf_test))
            transformed_test.append(
                _DataInfo(
                    test=data_info.test,
                    dataset_name=data_info.dataset_name,
                    transform_name=pre_model.name,
                    split_info=data_info.split_info,
                    scaler="None" if scaler is None else scaler.__class__.__name__,
                )
            )

    # ============================= inprocess models on transformed ===============================
    if fair_pipeline:
        run_on_transformed = inprocess_models
    else:
        # if not fair pipeline, run only the non-fair models on the transformed data
        run_on_transformed = [model for model in inprocess_models if not model.is_fairness_algo]

    transf_preds = await run_in_parallel(run_on_transformed, transformed_data, max_parallel)
    transf_results = _gather_metrics(
        transf_preds, transformed_test, run_on_transformed, metrics, per_sens_metrics, outdir, topic
    )
    all_results.append_df(transf_results)

    # ======================================== return all =========================================
    return all_results.results


def _gather_metrics(
    all_predictions: List[List[Prediction]],
    test_data: Sequence[_DataInfo],
    inprocess_models: Sequence[InAlgorithm],
    metrics: Sequence[Metric],
    per_sens_metrics: Sequence[Metric],
    outdir: Path,
    topic: Optional[str],
) -> Results:
    """Take a list of lists of predictions and compute all metrics."""
    columns = ["dataset", "scaler", "transform", "model", "split_id"]

    # transpose `all_results` so that the order in the results dataframe is correct
    num_cols = len(all_predictions[0]) if all_predictions else 0
    all_predictions_t = [[row[i] for row in all_predictions] for i in range(num_cols)]

    all_results = ResultsAggregator()

    # compute metrics, collect them and write them to files
    for preds_for_dataset, data_info in zip(all_predictions_t, test_data):
        # ============================= handle results of one dataset =============================
        results_df = pd.DataFrame(columns=columns)  # create empty results dataframe
        predictions: Prediction
        for predictions, model in zip(preds_for_dataset, inprocess_models):
            # construct a row of the results dataframe
            df_row: Dict[str, Union[str, float]] = {
                "dataset": data_info.dataset_name,
                "scaler": data_info.scaler,
                "transform": data_info.transform_name,
                "model": model.name,
                **data_info.split_info,
            }
            df_row.update(run_metrics(predictions, data_info.test, metrics, per_sens_metrics))

            results_df = results_df.append(df_row, ignore_index=True, sort=False)

        # write results to CSV files and load previous results from the files if they already exist
        csv_file = _result_path(outdir, data_info.dataset_name, data_info.transform_name, topic)
        aggregator = ResultsAggregator(results_df)
        # put old results before new results -> prepend=True
        aggregator.append_from_csv(csv_file, prepend=True)
        aggregator.save_as_csv(csv_file)
        all_results.append_df(aggregator.results)

    return all_results.results
