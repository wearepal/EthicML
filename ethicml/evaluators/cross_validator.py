import itertools
from typing import Dict, List, Tuple, Any

from ethicml.algorithms.algorithm_base import Algorithm
from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.utils import DataTuple
from typing import Type, TypeVar

from ethicml.metrics import Accuracy

U = TypeVar('U', bound=Algorithm)


class Results(object):

    def __init__(self):
        self.storage: List[Tuple[str, Any]] = []

    def append(self, config, score):
        self.storage.append([config, score])


class CrossValidator():
    def __init__(self, model: Type[U], hyperparams: Dict[str, List[float]]):
        self.model = model
        self.hyperparams = hyperparams
        self.results = Results()

        keys, values = zip(*hyperparams.items())
        self.experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def run(self, train: DataTuple):
        for experiment in self.experiments:
            m = self.model(hyperparams=experiment)
            preds = m.run(train, train)
            self.results.append(experiment, Accuracy().score(preds, train))

    def best(self) -> InAlgorithm:
        pass
