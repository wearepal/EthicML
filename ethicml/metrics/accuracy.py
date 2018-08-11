from sklearn.metrics import accuracy_score

from ethicml.metrics.metric import Metric
import pandas as pd


class Accuracy(Metric):
    def score(self, prediction, actual) -> float:
        if isinstance(actual, pd.DataFrame):
            actual = actual.values.ravel()

        return accuracy_score(actual, prediction)

    def get_name(self) -> str:
        return "Accuracy"
