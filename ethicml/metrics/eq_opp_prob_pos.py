"""
Used for claculating the probability of being poisitive given the class label is one... performs the
same as TPR per class, but does it in a different way.
"""


from ..algorithms.utils import DataTuple

import numpy
import pandas

from ethicml.algorithms.utils import make_dict
from ethicml.metrics.confusion_matrix import confusion_matrix
from ethicml.metrics.metric import Metric


def pos_subset_test(preds: numpy.array, data: Dict[str, pandas.DataFrame]):
    """Subset of the predictions that have a positive label associated with them

    Args:
        preds: predictions
        data: data including the labels

    Returns:
        subset of the supplied predictions
    """
    class_label = data['y']

    return preds[class_label == 1]


def pos_subset_data(data: Dict[str, pandas.DataFrame]):
    """Subset of the data that have a positive label associated with them

    Args:
        data: dictionary of DataFrames

    Returns:
        subset of the supplied data
    """
    features = data['x']
    sensitive_labels = data['s']
    class_labels = data['y']

    pos_x = features[class_labels == 1]
    pos_s = sensitive_labels[class_labels == 1]
    pos_y = class_labels[class_labels == 1]

    return make_dict(pos_x, pos_s, pos_y)


class EqOppProbPos(Metric):
    """Equality of Opportunity"""
    def score(self, prediction: numpy.array, actual: Dict[str, pandas.DataFrame]) -> float:
        pos_subset = pos_subset_data(actual)
        test_pos_subset = pos_subset_test(prediction, actual)

        _, f_pos, _, t_pos = confusion_matrix(test_pos_subset, pos_subset)

        return (t_pos + f_pos) / test_pos_subset.size

    @property
    def name(self) -> str:
        return "ProbPos | Y=1"

    @property
    def apply_per_sensitive(self) -> bool:
        return True
