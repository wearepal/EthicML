"""
For assessing PPV
"""

from ethicml.utility.data_structures import DataTuple, Predictions
from .confusion_matrix import confusion_matrix
from .metric import Metric


class PPV(Metric):
    """Positive predictive value"""

    def score(self, prediction: Predictions, actual: DataTuple) -> float:
        _, f_pos, _, t_pos = confusion_matrix(prediction, actual, self.positive_class)

        return t_pos / (t_pos + f_pos)

    @property
    def name(self) -> str:
        return "PPV"
