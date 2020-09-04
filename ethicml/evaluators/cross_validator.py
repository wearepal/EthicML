"""Cross Validation for any in process (at the moment) Algorithm."""
from collections import defaultdict
from itertools import product
from statistics import mean
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Type

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.cv import AbsCV
from ethicml.metrics.metric import Metric
from ethicml.preprocessing.train_test_split import fold_data
from ethicml.utility import DataTuple, Prediction, TrainTestPair

from .parallelism import run_in_parallel

__all__ = ["CrossValidator", "CVResults"]


class ResultTuple(NamedTuple):
    """Result of one experiment."""

    params: Dict[str, Any]  # the parameter setting used for this run
    fold_id: int  # the ID of the fold
    scores: Dict[str, float]  # the achieved scores


class CVResults:
    """Stores the results of a cross validation experiment."""

    def __init__(self, results: List[ResultTuple], model: Type[InAlgorithm]):
        self.raw_storage = results
        self.model = model
        self.mean_storage = self._organize_and_compute_means()

    def _organize_and_compute_means(self) -> Dict[str, ResultTuple]:
        """Compute means over folds and generate unique string for each hyperparameter setting."""
        # first, group the entries that have the same hyperparameters
        max_fold_id = 0
        grouped: Dict[str, List[ResultTuple]] = defaultdict(list)
        for result in self.raw_storage:
            # we convert the hyperparameter dictionaries to strings in order to compare them
            hyp_string = ", ".join(f"{key!s}={val!r}" for (key, val) in result.params.items())
            grouped[hyp_string].append(result)
            if result.fold_id > max_fold_id:
                max_fold_id = result.fold_id

        # compute the mean value of each measure within each group
        mean_vals: Dict[str, ResultTuple] = {}
        for hyp_string, results in grouped.items():
            # re-order the data: we want a dictionary that has all the scores as a list
            # first we create empty list according to the dictionary keys in the first result
            scores_dict: Dict[str, List[float]] = {
                score_name: [] for score_name in results[0].scores
            }
            # then we iterate over results and score names to fill the lists of score values
            for result in results:
                for score_name, score in result.scores.items():
                    scores_dict[score_name].append(score)

            # the re-ordering is complete. now we can compute the mean
            mean_dict = {score_name: mean(scores) for score_name, scores in scores_dict.items()}
            # we save everything as a ResultTuple. as we don't have a fold id, we use -1
            mean_vals[hyp_string] = ResultTuple(results[0].params, -1, mean_dict)

        assert len(mean_vals) * (max_fold_id + 1) == len(self.raw_storage)
        assert len(list(mean_vals.values())[-1].scores) == len(self.raw_storage[-1].scores)
        return mean_vals

    def get_best_result(self, measure: Metric) -> ResultTuple:
        """Get the hyperparameter combination for the best performance of a measure."""
        mean_vals = self.mean_storage

        def _get_score(item: Tuple[str, ResultTuple]) -> float:
            """Take an entry from `mean_storage` and return the desired score `measure`."""
            _, result = item
            return result.scores[measure.name]

        # find the best entry in `mean_storage` according to `measure`
        best_hyp_string, _ = max(mean_vals.items(), key=_get_score)

        return mean_vals[best_hyp_string]

    def best_hyper_params(self, measure: Metric) -> Dict[str, Any]:
        """Get best hyper-params."""
        return self.get_best_result(measure).params

    def best(self, measure: Metric) -> InAlgorithm:
        """Get best model."""
        return self.model(**self.best_hyper_params(measure))  # type: ignore[call-arg]

    def get_best_in_top_k(self, primary: Metric, secondary: Metric, top_k: int) -> ResultTuple:
        """Get best result in top K entries.

        First sort the results according to the primary metric, then take the best according to the
        secondary metric from the top K.
        """
        mean_vals = self.mean_storage

        def _get_primary_score(item: Tuple[str, ResultTuple]) -> float:
            return item[1].scores[primary.name]

        sorted_by_primary = sorted(mean_vals.items(), key=_get_primary_score)
        top_k_candidates = sorted_by_primary[:top_k]

        def _get_secondary_score(item: Tuple[str, ResultTuple]) -> float:
            return item[1].scores[secondary.name]

        best_hyp_string, _ = max(top_k_candidates, key=_get_secondary_score)
        return mean_vals[best_hyp_string]


class _ResultsAccumulator:
    def __init__(self, measures: Optional[List[Metric]] = None):
        self._measures = [Accuracy(), AbsCV()] if measures is None else measures
        self.results: List[ResultTuple] = []

    def __call__(
        self, parameter_setting: Dict[str, Any], preds: Prediction, test: DataTuple, fold_id: int
    ) -> Dict[str, float]:
        """Compute the scores for the given predictions and append to the list of results."""
        # compute all measures
        # TODO: this should also compute diffs and ratios
        scores = {measure.name: measure.score(preds, test) for measure in self._measures}
        # store the result
        self.results.append(ResultTuple(parameter_setting, fold_id, scores))
        return scores


class CrossValidator:
    """Object used to run cross-validation on a model."""

    def __init__(
        self,
        model: Type[InAlgorithm],
        hyperparams: Mapping[str, Sequence[Any]],
        folds: int = 3,
        max_parallel: int = 0,
    ):
        """The constructor takes the following arguments.

        Args:
            model: the class (not an instance) of the model for cross validation
            hyperparams: a dictionary where the keys are the names of hyperparameters and the values
                         are lists of possible values for the hyperparameters
            folds: the number of folds
            max_parallel: the maximum number of parallel processes; if set to 0, use the default
                          which is the number of available CPUs
        """
        self.model = model
        self.hyperparams = hyperparams
        self.folds = folds
        self.max_parallel = max_parallel

        keys, values = zip(*hyperparams.items())
        self.experiments: List[Dict[str, Any]] = [dict(zip(keys, v)) for v in product(*values)]

    async def run_async(
        self, train: DataTuple, measures: Optional[List[Metric]] = None
    ) -> CVResults:
        """Run the cross validation experiments asynchronously."""
        compute_scores_and_append = _ResultsAccumulator(measures)
        # instantiate all models
        models = [
            self.model(**experiment) for experiment in self.experiments  # type: ignore[call-arg]
        ]
        # create all folds
        data_folds: List[Tuple[DataTuple, DataTuple]] = list(fold_data(train, folds=self.folds))
        # convert to right format
        pair_folds = [TrainTestPair(train_fold, val) for (train_fold, val) in data_folds]
        # run everything in parallel
        all_results = await run_in_parallel(models, pair_folds, self.max_parallel)

        # finally, iterate over all results, compute scores and store them
        for preds_for_dataset, experiment in zip(all_results, self.experiments):
            for i, (preds, (_, val)) in enumerate(zip(preds_for_dataset, data_folds)):
                compute_scores_and_append(experiment, preds, val, i)
        return CVResults(compute_scores_and_append.results, self.model)

    def run(self, train: DataTuple, measures: Optional[List[Metric]] = None) -> CVResults:
        """Run the cross validation experiments."""
        compute_scores_and_append = _ResultsAccumulator(measures)
        for i, (train_fold, val) in enumerate(fold_data(train, folds=self.folds)):
            # run the models one by one and *immediately* report the scores on the measures
            for experiment in self.experiments:
                # instantiate model and run it
                model = self.model(**experiment)  # type: ignore[call-arg]
                preds = model.run(train_fold, val)
                scores = compute_scores_and_append(experiment, preds, val, i)
                score_string = ", ".join(f"{k}={v:.4g}" for k, v in scores.items())
                print(f"fold: {i}, model: '{model.name}', {score_string}, completed!")
        return CVResults(compute_scores_and_append.results, self.model)
