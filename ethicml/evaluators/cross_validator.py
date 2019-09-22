"""Cross Validation for any in process (at the moment) Algorithm"""
from collections import defaultdict
from itertools import product
from statistics import mean
from typing import Dict, List, Tuple, Any, Type, NamedTuple, Optional

import pandas as pd

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.utility.data_structures import DataTuple, TrainTestPair
from ethicml.metrics import Accuracy, Metric, AbsCV
from ethicml.preprocessing.train_test_split import fold_data
from .parallelism import run_in_parallel


class ResultTuple(NamedTuple):
    """Result of one experiment"""

    params: Dict[str, Any]
    fold_id: int
    scores: Dict[str, float]


class CVResults:
    """
    Stores the results of a cross validation experiment
    """

    def __init__(self, results: List[ResultTuple], model: Type[InAlgorithm]):
        self.raw_storage = results
        self.model = model
        self.mean_storage = self._organize_and_compute_means()

    def _organize_and_compute_means(self) -> Dict[str, ResultTuple]:
        """Compute means over folds and generate unique string for each hyperparameter setting"""
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
        """
        Get the hyperparameter combination for the best performance of a measure
        """
        mean_vals = self.mean_storage

        def _get_score(item: Tuple[str, ResultTuple]) -> float:
            """Take an entry from `mean_storage` and return the desired score `measure`"""
            _, result = item
            return result.scores[measure.name]

        # find the best entry in `mean_storage` according to `measure`
        best_hyp_string, _ = max(mean_vals.items(), key=_get_score)

        return mean_vals[best_hyp_string]

    def best_hyper_params(self, measure: Metric) -> Dict[str, Any]:
        return self.get_best_result(measure).params

    def best(self, measure: Metric) -> InAlgorithm:
        return self.model(**self.best_hyper_params(measure))

    def get_best_in_top_k(self, primary: Metric, secondary: Metric, top_k: int) -> ResultTuple:
        """
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


class CrossValidator:
    """
    Object used to run cross-validation on a model
    """

    def __init__(
        self,
        model: Type[InAlgorithm],
        hyperparams: Dict[str, List[Any]],
        folds: int = 3,
        parallel: bool = False,
        max_parallel: int = 0,
    ):
        """
        Args:
            model: the class (not an instance) of the model for cross validation
            hyperparams: a dictionary where the keys are the names of hyperparameters and the values
                         are lists of possible values for the hyperparameters
            folds: the number of folds
            parallel: if True, run the algorithms in parallel
            max_parallel: the maximum number of parallel processes; if set to 0, use the default
                          which is the number of available CPUs
        """
        self.model = model
        self.hyperparams = hyperparams
        self.folds = folds
        self.parallel = parallel
        self.max_parallel = max_parallel

        keys, values = zip(*hyperparams.items())
        self.experiments: List[Dict[str, Any]] = [dict(zip(keys, v)) for v in product(*values)]

    def run(self, train: DataTuple, measures: Optional[List[Metric]] = None) -> CVResults:
        """Run the cross validation experiments"""
        measures_ = [Accuracy(), AbsCV()] if measures is None else measures

        results: List[ResultTuple] = []

        def _compute_scores_and_append(
            experiment: Dict[str, Any], preds: pd.DataFrame, test: DataTuple, fold_id: int
        ) -> Dict[str, float]:
            # compute all measures
            # TODO: this should also compute diffs and ratios
            scores = {measure.name: measure.score(preds, test) for measure in measures_}
            # store the result
            results.append(ResultTuple(experiment, fold_id, scores))
            return scores

        if self.parallel:
            # instantiate all models
            models = [self.model(**experiment) for experiment in self.experiments]
            # create all folds
            data_folds: List[Tuple[DataTuple, DataTuple]] = list(fold_data(train, folds=self.folds))
            # convert to right format
            pair_folds = [TrainTestPair(train_fold, val) for (train_fold, val) in data_folds]
            # run everything in parallel
            all_results = run_in_parallel(models, pair_folds, self.max_parallel)

            # finally, iterate over all results, compute scores and store them
            for preds_for_dataset, experiment in zip(all_results, self.experiments):
                for i, (preds, (train_fold, val)) in enumerate(zip(preds_for_dataset, data_folds)):
                    _compute_scores_and_append(experiment, preds, val, i)
        else:
            for i, (train_fold, val) in enumerate(fold_data(train, folds=self.folds)):
                # run the models one by one and *immediately* report the scores on the measures
                for experiment in self.experiments:
                    # instantiate model and run it
                    model = self.model(**experiment)
                    preds = model.run(train_fold, val)
                    scores = _compute_scores_and_append(experiment, preds, val, i)
                    score_string = ", ".join(f"{k}={v:.4g}" for k, v in scores.items())
                    print(f"fold: {i}, model: '{model.name}', {score_string}, completed!")
        return CVResults(results, self.model)
