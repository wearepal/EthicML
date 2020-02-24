"""For assessing PPV."""

from ethicml.common import implements
from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import Metric


class PPV(Metric):
    """Positive predictive value."""

    _name: str = "PPV"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        _, f_pos, _, t_pos = confusion_matrix(prediction, actual, self.positive_class)

        return t_pos / (t_pos + f_pos)
