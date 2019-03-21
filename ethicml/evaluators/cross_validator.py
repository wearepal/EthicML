import itertools
from typing import Dict, List, Tuple, Any

from ethicml.algorithms.algorithm_base import Algorithm
from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.utils import DataTuple
from typing import Type, TypeVar
from operator import itemgetter

from ethicml.metrics import Accuracy, Metric, CV
from ethicml.preprocessing.train_test_split import train_test_split
from itertools import groupby


U = TypeVar('U', bound=Algorithm)


class Results(object):

    def __init__(self):
        self.storage: List[Tuple[str, str, Any, int]] = []

    def append(self, config, score_name, score, fold_id):
        self.storage.append((config, score_name, score, fold_id))

    def get_params_for_best(self, measure):
        candidates = [val for val in self.storage if val[1] == measure.name]

        glo = [list(y) for x, y in groupby(candidates, key=itemgetter(0))]

        for key in candidates[:,:1]:
            print(key)
        return max(candidates, key=itemgetter(2))[0]


class CrossValidator():
    def __init__(self, model: Type[U], hyperparams: Dict[str, List[float]], folds: int = 3):
        self.model = model
        self.hyperparams = hyperparams
        self.results = Results()
        self.folds = folds

        keys, values = zip(*hyperparams.items())
        self.experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def run(self, train: DataTuple, measures=None):
        if measures is None:
            measures = [Accuracy(), CV()]

        for i in range(self.folds):
            train, val = train_test_split(train, random_seed=i)
            for experiment in self.experiments:
                m = self.model(hyperparams=experiment)
                preds = m.run(train, train)
                for measure in measures:
                    self.results.append(experiment, measure.name, measure.score(preds, train), i)

    def best(self, measure) -> InAlgorithm:
        return self.model(hyperparams=self.results.get_params_for_best(measure))
