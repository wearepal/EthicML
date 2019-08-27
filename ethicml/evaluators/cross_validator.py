"""Cross Validation for any in process (at the moment) Algorithm"""


from collections import defaultdict
from itertools import product
from statistics import mean
from typing import Dict, List, Tuple, Any, Type, NamedTuple, Optional

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.utility.data_structures import DataTuple
from ethicml.metrics import Accuracy, Metric, CV
from ethicml.preprocessing.train_test_split import fold_data


class ResultTuple(NamedTuple):
    """Result of one experiment"""

    params: Dict[str, Any]
    fold_id: int
    scores: Dict[str, float]


class Results:
    """
    Stores the results of the experiments
    """

    def __init__(self):
        self.raw_storage: List[ResultTuple] = []
        self.mean_storage: Optional[Dict[str, ResultTuple]] = None

    def append(self, config: Dict[str, Any], fold_id: int, scores: Dict[str, float]) -> None:
        self.raw_storage.append(ResultTuple(config, fold_id, scores))

    def _organize_and_compute_means(self) -> Dict[str, ResultTuple]:
        """Compute means over folds and generate unique string for each hyperparameter setting"""
        # check whether we have already done this
        if self.mean_storage is not None:
            return self.mean_storage

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
        # store the mean values for next time
        self.mean_storage = mean_vals
        return mean_vals

    def get_best_result(self, measure: Metric) -> ResultTuple:
        """
        Get the hyperparameter combination for the best performance of a measure
        """
        mean_vals = self._organize_and_compute_means()

        def _get_score(item: Tuple[str, ResultTuple]):
            _, result = item
            return result.scores[measure.name]

        best_hyp_string, _ = max(mean_vals.items(), key=_get_score)

        return mean_vals[best_hyp_string]

    def get_best_in_top_k(self, primary: Metric, secondary: Metric, top_k: int) -> ResultTuple:
        """
        First sort the results according to the primary metric, then take the best according to the
        secondary metric from the top K.
        """
        mean_vals = self._organize_and_compute_means()

        def _get_primary_score(item: Tuple[str, ResultTuple]):
            return item[1].scores[primary.name]

        sorted_by_primary = sorted(mean_vals.items(), key=_get_primary_score)
        top_k_candidates = sorted_by_primary[:top_k]

        def _get_secondary_score(item: Tuple[str, ResultTuple]):
            return item[1].scores[secondary.name]

        best_hyp_string, _ = max(top_k_candidates, key=_get_secondary_score)
        return mean_vals[best_hyp_string]


class CrossValidator:
    """
    Object used to run cross-validation on a model
    """

    def __init__(self, model: Type[InAlgorithm], hyperparams: Dict[str, List[Any]], folds: int = 3):
        self.model = model
        self.hyperparams = hyperparams
        self.results = Results()
        self.folds = folds

        keys, values = zip(*hyperparams.items())
        self.experiments = [dict(zip(keys, v)) for v in product(*values)]

    def run(self, train: DataTuple, measures: Optional[List[Metric]] = None) -> None:
        """runs the cross validation experiments"""
        if measures is None:
            measures = [Accuracy(), CV()]

        for i, (train_fold, val) in enumerate(fold_data(train, folds=self.folds)):
            for experiment in self.experiments:
                # instantiate model and run it
                model = self.model(**experiment)
                preds = model.run(train_fold, val)
                # compute all measures
                scores = {measure.name: measure.score(preds, val) for measure in measures}
                # store the result
                self.results.append(experiment, i, scores)
                score_string = ", ".join(f"{k}={v:.4g}" for k, v in scores.items())
                print(f"fold: {i}, model: '{model.name}', {score_string}, completed!")

    def best(self, measure: Metric) -> InAlgorithm:
        return self.model(**self.best_hyper_params(measure))

    def best_hyper_params(self, measure: Metric) -> Dict[str, Any]:
        return self.results.get_best_result(measure).params
