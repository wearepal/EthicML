"""Cross Validation for any in process (at the moment) Algorithm."""
from __future__ import annotations
from collections import defaultdict
from itertools import product
from statistics import mean
from typing import Any, Mapping, NamedTuple, Sequence, Type

from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.cv import AbsCV
from ethicml.metrics.metric import Metric
from ethicml.models.inprocess.in_algorithm import InAlgorithm
from ethicml.preprocessing.splits import fold_data
from ethicml.run.parallelism import run_in_parallel
from ethicml.utility import DataTuple, Prediction, TrainValPair

__all__ = ["CrossValidator", "CVResults"]


class ResultTuple(NamedTuple):
    """Result of one experiment."""

    params: dict[str, Any]  # the parameter setting used for this run
    fold_id: int  # the ID of the fold
    scores: dict[str, float]  # the achieved scores


class CVResults:
    """Stores the results of a cross validation experiment (see :class:`CrossValidator`).

    This object isn't meant to be iterated over directly.
    Instead, use the ``raw_storage`` property to access the results across all folds.
    Or, use the ``mean_storage`` property to access the average results for each parameter setting.


    .. code-block:: python

        import ethicml as em
        from ethicml import data, metrics, models
        from ethicml.run import CrossValidator

        train, test = em.train_test_split(data.Compas().load())
        hyperparams = {"C": [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}

        cv = CrossValidator(models.LR, hyperparams, folds=3)
        primary = metrics.Accuracy()
        fair_measure = metrics.AbsCV()
        cv_results = cv.run(train, measures=[primary, fair_measure])
        best_result = cv_results.get_best_in_top_k(primary, fair_measure, top_k=3)

        print(f"Best C: {best_result.params['C']}")
        print(f"Best Accuracy: {best_result.scores['Accuracy']}")
        print(f"Best CV Score: {best_result.scores['CV absolute']}")
        print(cv_results.mean_storage)
        print(cv_results.raw_storage)

    """

    def __init__(self, results: list[ResultTuple], model: type[InAlgorithm]):
        self.raw_storage = results
        self.model = model
        self.mean_storage = self._organize_and_compute_means()

    def _organize_and_compute_means(self) -> dict[str, ResultTuple]:
        """Compute means over folds and generate unique string for each hyperparameter setting."""
        # first, group the entries that have the same hyperparameters
        max_fold_id = 0
        grouped: dict[str, list[ResultTuple]] = defaultdict(list)
        for result in self.raw_storage:
            # we convert the hyperparameter dictionaries to strings in order to compare them
            hyp_string = ", ".join(f"{key!s}={val!r}" for (key, val) in result.params.items())
            grouped[hyp_string].append(result)
            if result.fold_id > max_fold_id:
                max_fold_id = result.fold_id

        # compute the mean value of each measure within each group
        mean_vals: dict[str, ResultTuple] = {}
        for hyp_string, results in grouped.items():
            # re-order the data: we want a dictionary that has all the scores as a list
            # first we create empty list according to the dictionary keys in the first result
            scores_dict: dict[str, list[float]] = {
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

        def _get_score(item: tuple[str, ResultTuple]) -> float:
            """Take an entry from `mean_storage` and return the desired score `measure`."""
            _, result = item
            return result.scores[measure.name]

        # find the best entry in `mean_storage` according to `measure`
        best_hyp_string, _ = max(mean_vals.items(), key=_get_score)

        return mean_vals[best_hyp_string]

    def best_hyper_params(self, measure: Metric) -> dict[str, Any]:
        """Get hyper-parameters that return the 'best' result for the metric of interest."""
        return self.get_best_result(measure).params

    def best(self, measure: Metric) -> InAlgorithm:
        """Return a model initialised with the hyper-parameters that perform optimally on average across folds for a given metric."""
        return self.model(**self.best_hyper_params(measure))

    def get_best_in_top_k(self, primary: Metric, secondary: Metric, top_k: int) -> ResultTuple:
        """Get best result in top-K entries.

        First sort the results according to the primary metric, then take the best according to the
        secondary metric from the top K.

        :param primary: Metric to first sort by.
        :param secondary: Metric to sort the top-K models by for a second time, the top will be
            selected.
        :param top_k: Number of entries to consider.
        :returns: A tuple with the parameters, the fold ID and the scores.
        """
        mean_vals = self.mean_storage

        def _get_primary_score(item: tuple[str, ResultTuple]) -> float:
            return item[1].scores[primary.name]

        sorted_by_primary = sorted(mean_vals.items(), key=_get_primary_score)
        top_k_candidates = sorted_by_primary[:top_k]

        def _get_secondary_score(item: tuple[str, ResultTuple]) -> float:
            return item[1].scores[secondary.name]

        best_hyp_string, _ = max(top_k_candidates, key=_get_secondary_score)
        return mean_vals[best_hyp_string]


class _ResultsAccumulator:
    def __init__(self, measures: list[Metric] | None = None):
        self._measures = [Accuracy(), AbsCV()] if measures is None else measures
        self.results: list[ResultTuple] = []

    def __call__(
        self, parameter_setting: dict[str, Any], preds: Prediction, test: DataTuple, fold_id: int
    ) -> dict[str, float]:
        """Compute the scores for the given predictions and append to the list of results."""
        # compute all measures
        # TODO: this should also compute diffs and ratios
        scores = {measure.name: measure.score(preds, test) for measure in self._measures}
        # store the result
        self.results.append(ResultTuple(parameter_setting, fold_id, scores))
        return scores


class CrossValidator:
    """A simple approach to Cross Validation.

    The CrossValidator object is used to run cross-validation on a model. Results are returned in
    a :class:`CVResults` object.

    .. code-block:: python

        import ethicml as em
        from ethicml import data, metrics, models
        from ethicml.run import CrossValidator

        train, test = em.train_test_split(data.Compas().load())
        hyperparams = {"C": [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}

        lr_cv = CrossValidator(models.LR, hyperparams, folds=3)

        primary = metrics.Accuracy()
        fair_measure = metrics.AbsCV()
        cv_results = lr_cv.run(train, measures=[primary, fair_measure])

    :param model: the class (not an instance) of the model for cross validation
    :param hyperparams: a dictionary where the keys are the names of hyperparameters and the values
        are lists of possible values for the hyperparameters
    :param folds: the number of folds
    :param max_parallel: the maximum number of parallel processes; if set to 0, use the default
        which is the number of available CPUs
    """

    def __init__(
        self,
        model: Type[InAlgorithm],
        hyperparams: Mapping[str, Sequence[Any]],
        folds: int = 3,
        max_parallel: int = 0,
    ):
        self.model = model
        self.hyperparams = hyperparams
        self.folds = folds
        self.max_parallel = max_parallel

        keys, values = zip(*hyperparams.items())
        self.experiments: list[dict[str, Any]] = [dict(zip(keys, v)) for v in product(*values)]

    def run_async(self, train: DataTuple, measures: list[Metric] | None = None) -> CVResults:
        """Run the cross validation experiments asynchronously.

        :param train: the training data
        :param measures:  (Default: None)
        :returns: CVResults
        """
        compute_scores_and_append = _ResultsAccumulator(measures)
        # instantiate all models
        models = [self.model(**experiment) for experiment in self.experiments]
        # create all folds
        data_folds: list[tuple[DataTuple, DataTuple]] = list(fold_data(train, folds=self.folds))
        # convert to right format
        pair_folds = [TrainValPair(train_fold, val) for (train_fold, val) in data_folds]
        # run everything in parallel
        all_results = run_in_parallel(
            algos=models,
            data=pair_folds,
            seeds=[0] * len(pair_folds),
            num_jobs=self.max_parallel,
        )

        # finally, iterate over all results, compute scores and store them
        for preds_for_dataset, experiment in zip(all_results, self.experiments):
            for i, (preds, (_, val)) in enumerate(zip(preds_for_dataset, data_folds)):
                compute_scores_and_append(experiment, preds, val, i)
        return CVResults(compute_scores_and_append.results, self.model)

    def run(self, train: DataTuple, measures: list[Metric] | None = None) -> CVResults:
        """Run the cross validation experiments."""
        compute_scores_and_append = _ResultsAccumulator(measures)
        for (i, (train_fold, val)), experiment in product(
            enumerate(fold_data(train, folds=self.folds)), self.experiments
        ):
            # instantiate model and run it
            model = self.model(**experiment)
            preds = model.run(train_fold, val)
            scores = compute_scores_and_append(experiment, preds, val, i)
            score_string = ", ".join(f"{k}={v:.4g}" for k, v in scores.items())
            print(f"fold: {i}, model: '{model.name}', {score_string}, completed!")
        return CVResults(compute_scores_and_append.results, self.model)
