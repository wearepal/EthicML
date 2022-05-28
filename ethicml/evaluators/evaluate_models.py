"""Runs given metrics on given algorithms for given datasets."""
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Union
from typing_extensions import Literal

import pandas as pd

from ethicml.data.dataset import Dataset
from ethicml.data.load import load_data
from ethicml.evaluators.parallelism import run_in_parallel
from ethicml.metrics.metric import Metric
from ethicml.metrics.per_sensitive_attribute import (
    MetricNotApplicable,
    diff_per_sensitive_attribute,
    metric_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
)
from ethicml.models.inprocess.in_algorithm import InAlgorithm
from ethicml.models.preprocess.pre_algorithm import PreAlgorithm
from ethicml.preprocessing.scaling import ScalerType, scale_continuous
from ethicml.preprocessing.splits import DataSplitter, RandomSplit
from ethicml.utility.data_structures import (
    DataTuple,
    EvalTuple,
    HyperParamValue,
    Prediction,
    Results,
    ResultsAggregator,
    TrainValPair,
    make_results,
)

__all__ = ["evaluate_models", "run_metrics", "load_results"]


def get_sensitive_combinations(metrics: List[Metric], train: DataTuple) -> List[str]:
    """Get all possible combinations of sensitive attribute and metrics.

    :param metrics:
    :param train:
    """
    poss_values = [f"{train.s.name}_{unique}" for unique in train.s.unique()]
    return [f"{s}_{m.name}" for s in poss_values for m in metrics]


def per_sens_metrics_check(per_sens_metrics: Sequence[Metric]) -> None:
    """Check if the given metrics allow application per sensitive attribute.

    :param per_sens_metrics:
    """
    for metric in per_sens_metrics:
        if not metric.apply_per_sensitive:
            raise MetricNotApplicable(
                f"Metric {metric.name} is not applicable per sensitive "
                f"attribute, apply to whole dataset instead"
            )


def run_metrics(
    predictions: Prediction,
    actual: EvalTuple,
    metrics: Sequence[Metric] = (),
    per_sens_metrics: Sequence[Metric] = (),
    diffs_and_ratios: bool = True,
    use_sens_name: bool = True,
) -> Dict[str, HyperParamValue]:
    """Run all the given metrics on the given predictions and return the results.

    :param predictions: DataFrame with predictions
    :param actual: EvalTuple with the labels
    :param metrics: list of metrics (Default: ())
    :param per_sens_metrics: list of metrics that are computed per sensitive attribute (Default: ())
    :param diffs_and_ratios: if True, compute diffs and ratios per sensitive attribute (Default: True)
    :param use_sens_name: if True, use the name of the senisitive variable in the returned results.
                        If False, refer to the sensitive varibale as `S`. (Default: True)
    """
    result: Dict[str, HyperParamValue] = {}
    if predictions.hard.isna().any(axis=None):  # type: ignore[arg-type]
        return {"algorithm_failed": 1.0}
    for metric in metrics:
        result[metric.name] = metric.score(predictions, actual)

    for metric in per_sens_metrics:
        per_sens = metric_per_sensitive_attribute(predictions, actual, metric, use_sens_name)
        if diffs_and_ratios:
            diff_per_sens = diff_per_sensitive_attribute(per_sens)
            ratio_per_sens = ratio_per_sensitive_attribute(per_sens)
            per_sens.update(diff_per_sens)
            per_sens.update(ratio_per_sens)
        for key, value in per_sens.items():
            result[f"{metric.name}_{key}"] = value
    result.update(predictions.info)
    return result  # SUGGESTION: we could return a DataFrame here instead of a dictionary


def load_results(
    dataset_name: str,
    transform_name: str,
    topic: Optional[str] = None,
    outdir: Path = Path(".") / "results",
) -> Optional[Results]:
    """Load results from a CSV file that was created by `evaluate_models`.

    :param dataset_name: name of the dataset of the results
    :param transform_name: name of the transformation that was used for the results
    :param topic: (optional) topic string of the results (Default: None)
    :param outdir: directory where the results are stored (Default: Path(".") / "results")
    :returns: DataFrame if the file exists; None otherwise
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
        transform_list.extend(preprocess_model.name for preprocess_model in transforms)
        for transform_name in transform_list:
            path_to_file: Path = _result_path(outdir, dataset.name, transform_name, topic)
            if path_to_file.exists():
                path_to_file.unlink()


class _DataInfo(NamedTuple):
    test: DataTuple
    dataset_name: str
    transform_name: str
    split_info: Dict[str, float]
    scaler: str


def evaluate_models(
    datasets: List[Dataset],
    *,
    preprocess_models: Sequence[PreAlgorithm] = (),
    inprocess_models: Sequence[InAlgorithm] = (),
    metrics: Sequence[Metric] = (),
    per_sens_metrics: Sequence[Metric] = (),
    repeats: int = 1,
    test_mode: bool = False,
    delete_previous: bool = True,
    splitter: Optional[DataSplitter] = None,
    topic: Optional[str] = None,
    fair_pipeline: bool = True,
    num_jobs: Optional[int] = None,
    scaler: Optional[ScalerType] = None,
    repeat_on: Literal["data", "model", "both"] = "both",
) -> Results:
    """Evaluate all the given models for all the given datasets and compute all the given metrics.

    :param datasets: List of dataset objects.
    :param preprocess_models: List of preprocess model objects. (Default: ())
    :param inprocess_models: List of inprocess model objects. (Default: ())
    :param metrics: List of metric objects. (Default: ())
    :param per_sens_metrics: List of metric objects that will be evaluated per sensitive attribute.
        (Default: ())
    :param repeats: Number of repeats to perform for the experiments. (Default: 1)
    :param test_mode: If True, only use a small subset of the data so that the models run faster.
        (Default: False)
    :param delete_previous: True by default. If True, delete previous results in the directory.
    :param splitter: Custom train-test splitter. (Default: None)
    :param topic: A string that identifies the run; the string is prepended to the filename.
        (Default: None)
    :param fair_pipeline: if True, run fair inprocess algorithms on the output of preprocessing.
        (Default: True)
    :param num_jobs: Number of parallel jobs; if None, the number of CPUs is used. (Default: None)
    :param scaler: Sklearn-style scaler to be used on the continuous features. (Default: None)
    :param repeat_on: Should the `data` or `model` seed be varied for each run? Or should they `both` be the same? (Default: "both")
    """
    per_sens_metrics_check(per_sens_metrics)
    if splitter is None:
        if repeat_on == "model":
            train_test_split: DataSplitter = RandomSplit(train_percentage=0.8, start_seed=None)
        else:
            train_test_split = RandomSplit(train_percentage=0.8, start_seed=0)
    else:
        train_test_split = splitter

    default_transform_name = "no_transform"

    outdir = Path(".") / "results"  # OS-independent way of saying './results'
    outdir.mkdir(exist_ok=True)

    if delete_previous:
        _delete_previous_results(outdir, datasets, preprocess_models, topic)

    all_results = ResultsAggregator()

    # ======================================= prepare data ========================================
    data_splits: List[TrainValPair] = []
    test_data: List[_DataInfo] = []  # contains the test set and other things needed for the metrics
    model_seeds: List[int] = []
    for dataset in datasets:
        for split_id in range(repeats):
            train: DataTuple
            test: DataTuple
            train, test, split_info = train_test_split(load_data(dataset), split_id=split_id)
            if scaler is not None:
                train, scaler_post = scale_continuous(dataset, train, scaler)
                test, _ = scale_continuous(dataset, test, scaler_post, fit=False)
            if test_mode:
                # take smaller subset of training data to speed up training
                train = train.get_n_samples()
            train = train.rename(f"{train.name} ({split_id})")
            data_splits.append(TrainValPair(train, test))
            model_seeds.append(0 if repeat_on == "data" else split_id)
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

        # load previous results
        csv_file = _result_path(outdir, dataset.name, default_transform_name, topic)
        all_results.append_from_csv(csv_file)

    # ============================= inprocess models on untransformed =============================
    all_predictions = run_in_parallel(
        algos=inprocess_models, data=data_splits, seeds=model_seeds, num_jobs=num_jobs
    )
    inprocess_untransformed = _gather_metrics(
        all_predictions, test_data, inprocess_models, metrics, per_sens_metrics, outdir, topic
    )
    all_results.append_df(inprocess_untransformed)

    # ===================================== preprocess models =====================================
    # run all preprocess models
    all_transformed = run_in_parallel(
        algos=preprocess_models, data=data_splits, seeds=model_seeds, num_jobs=num_jobs
    )

    # append the transformed data to `transformed_data`
    transformed_data: List[TrainValPair] = []
    transformed_test: List[_DataInfo] = []
    for transformed, pre_model in zip(all_transformed, preprocess_models):
        for (transf_train, transf_test), data_info in zip(transformed, test_data):
            transformed_data.append(TrainValPair(transf_train, transf_test))
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

    transf_preds = run_in_parallel(
        algos=run_on_transformed,
        data=transformed_data,
        seeds=[0] * len(transformed_data),
        num_jobs=num_jobs,
    )
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
    """Take a list of lists of predictions and compute all metrics.

    :param all_predictions:
    :param test_data:
    :param inprocess_models:
    :param metrics:
    :param per_sens_metrics:
    :param outdir:
    :param topic:
    """
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
            hyperparameters: Dict[str, Union[str, float]] = {
                k: v if isinstance(v, (float, int)) else str(v)
                for k, v in model.get_hyperparameters().items()
            }

            seed = predictions.info["model_seed"]
            assert isinstance(seed, int)
            df_row: Dict[str, HyperParamValue] = {
                "dataset": data_info.dataset_name,
                "scaler": data_info.scaler,
                "transform": data_info.transform_name,
                "model": model.name,
                "model_seed": seed,
                **data_info.split_info,
                **hyperparameters,
            }
            df_row.update(run_metrics(predictions, data_info.test, metrics, per_sens_metrics))

            results_df = results_df.append(df_row, ignore_index=True, sort=False)

        # write results to CSV files and load previous results from the files if they already exist
        csv_file = _result_path(outdir, data_info.dataset_name, data_info.transform_name, topic)
        aggregator = ResultsAggregator(results_df)
        # put old results before new results -> prepend=True
        aggregator.append_from_csv(csv_file, prepend=True)
        aggregator.save_as_csv(csv_file)
        all_results.append_df(results_df)

    return all_results.results
