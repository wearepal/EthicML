""""
Cross Validation for any in process (at the moment) Algoruthm"""


from collections import defaultdict
from itertools import product
from operator import itemgetter
from statistics import mean
from typing import Dict, List, Tuple, Any
from typing import Type

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.utility.data_structures import DataTuple
from ethicml.metrics import Accuracy, Metric, CV
from ethicml.preprocessing.train_test_split import fold_data


class Results:
    """
    Stores the results of the experiments
    """

    def __init__(self):
        self.storage: List[Tuple[Dict[str, Any], int, str, float]] = []

    def append(self, config: Dict[str, Any], fold_id: int, score_name: str, score: float) -> None:
        self.storage.append((config, fold_id, score_name, score))

    def get_params_for_best(self, measure: Metric) -> Dict[str, Any]:
        """
        Get the hyperparameter combination for the best performance of a measure
        Args:
            measure:

        Returns:

        """
        candidates = [val for val in self.storage if val[2] == measure.name]

        grouped: Dict[str, List[Any]] = defaultdict(list)
        for tup in candidates:
            hyp_dic: Dict[str, Any] = tup[0]
            str_hyp_dict = ", ".join("{!s}={!r}".format(key, val) for (key, val) in hyp_dic.items())
            grouped[str_hyp_dict].append(tup[1:])

        mean_vals: Dict[str, float] = defaultdict(float)
        for exp, results in grouped.items():
            results = [list(res) for res in results]
            mean_vals[exp] = mean([res[2] for res in results])

        best_params = (max(mean_vals.items(), key=itemgetter(1)))[0]

        params_for_best = candidates[0][0]
        for k in candidates:
            hyp_dic = k[0]
            str_hyp_dict = ", ".join("{!s}={!r}".format(key, val) for (key, val) in hyp_dic.items())
            if str_hyp_dict == best_params:
                params_for_best = hyp_dic
                break
        return params_for_best


class CrossValidator:
    """
    Object used to run cross-validatioon on a model
    """

    def __init__(self, model: Type[InAlgorithm], hyperparams: Dict[str, List[Any]], folds: int = 3):
        self.model = model
        self.hyperparams = hyperparams
        self.results = Results()
        self.folds = folds

        keys, values = zip(*hyperparams.items())
        self.experiments = [dict(zip(keys, v)) for v in product(*values)]

    def run(self, train: DataTuple, measures=None):
        """
        runs the cross validation experiments
        Args:
            train:
            measures:

        Returns:

        """
        if measures is None:
            measures = [Accuracy(), CV()]

        for i, (train_fold, val) in enumerate(fold_data(train, folds=self.folds)):
            for experiment in self.experiments:
                model = self.model(**experiment)
                preds = model.run(train_fold, val)
                for measure in measures:
                    self.results.append(experiment, i, measure.name, measure.score(preds, val))
                print(f"fold_{i}_model_{model.name}_completed")

    def best(self, measure) -> InAlgorithm:
        return self.model(**self.results.get_params_for_best(measure))

    def best_hyper_params(self, measure):
        return self.results.get_params_for_best(measure)
