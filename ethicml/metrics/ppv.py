"""
For assessing PPV
"""

import numpy as np

from .confusion_matrix import confusion_matrix
from .metric import Metric
from ..algorithms.utils import DataTuple


class PPV(Metric):
    """Positive predictive value"""
    def score(self, prediction: np.array, actual: DataTuple) -> float:
        _, f_pos, _, t_pos = confusion_matrix(prediction, actual)

        return t_pos / (t_pos + f_pos)

    @property
    def name(self) -> str:
        return "PPV"
