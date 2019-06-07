"""
Implementation of the variant of toal variation used as the unfairness metric in
https://arxiv.org/abs/1905.13662
"""
import numpy as np
import pandas as pd

from ethicml.algorithms.utils import DataTuple
from ethicml.metrics import Metric
from ethicml.utility.heaviside import Heaviside


class TV(Metric):
    """
    Requires that predictions be probabilities of the class
    """

    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:

        # Only consider one s column (for now)
        s_col = actual.s.columns[0]
        unique_s = actual.s[s_col].unique()

        heavi = Heaviside()
        class_labels = heavi.apply(prediction.values)
        unique_y = np.unique(class_labels)
        class_labels = pd.DataFrame(class_labels, columns=["labels"])

        sum_dist = 0

        for y in unique_y:
            for s in unique_s:
                to_check = prediction.loc[actual.s[s_col] == s]
                to_check = to_check.loc[class_labels["labels"] == y]
                to_check = to_check.values.mean()

                avg_pred = prediction[class_labels["labels"] == y].values.mean()

                sum_dist += abs(avg_pred - to_check)

        return sum_dist / len(unique_s)

    @property
    def name(self) -> str:
        return "TV"
