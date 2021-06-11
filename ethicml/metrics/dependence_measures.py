"""For assessing the Mutual Information between s and yhat."""
from typing import ClassVar

import numpy as np
from kit import implements
from sklearn.metrics import normalized_mutual_info_score
from typing_extensions import Literal

from ethicml.utility import DataTuple, Prediction

from .metric import Metric

__all__ = ["NMI", "RenyiCorrelation", "Yanovich"]


class _DependenceMeasure(Metric):
    """Base class for dependence measures, which tell you how dependent two variables are."""

    _base_name: ClassVar[str]

    def __init__(self, pos_class: int = 1, base: Literal["y", "s"] = "s"):
        super().__init__(pos_class=pos_class)
        if base not in ["s", "y"]:
            raise NotImplementedError(
                "Can only calculate NMI of prediction.hards with regard to y or s"
            )
        self._name = f"{self._base_name} preds and {base}"
        self.base = base

    @property
    def apply_per_sensitive(self) -> bool:
        """Can this metric be applied per sensitive attribute group?"""
        return False


class NMI(_DependenceMeasure):
    """Normalized Mutual Information.

    Also called V-Measure. Defined in this paper: https://www.aclweb.org/anthology/D07-1043.pdf
    """

    _base_name: ClassVar[str] = "NMI"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        base_values = actual.y if self.base == "y" else actual.s
        return normalized_mutual_info_score(
            base_values.to_numpy().ravel(),
            prediction.hard.to_numpy().ravel(),
            average_method="arithmetic",
        )


class Yanovich(_DependenceMeasure):
    """Yanovich Metric. Measures how dependent two random variables are.

    As defined in this paper: https://arxiv.org/abs/1008.0492
    """

    _base_name: ClassVar[str] = "Yanovich"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        base_values = actual.y if self.base == "y" else actual.s
        return self._dependence(base_values.to_numpy().ravel(), prediction.hard.to_numpy().ravel())

    @staticmethod
    def _dependence(x: np.ndarray, y: np.ndarray) -> float:
        x_vals = np.unique(x)
        y_vals = np.unique(y)
        if len(x_vals) < 2 or len(y_vals) < 2:
            return 1.0

        total = len(x)
        assert total == len(y)

        joint = np.empty((len(x_vals), len(y_vals)))

        for i, x_val in enumerate(x_vals):
            for k, y_val in enumerate(y_vals):
                # count how often x_val and y_val co-occur
                joint[i, k] = _count_true((x == x_val) & (y == y_val)) / total

        sum_sq_determinants = 0.0  # the sum of the squares of all second-order determinants
        # find all 2x2 submatrices and compute their determinant
        for i in range(len(x_vals)):
            # i < j
            for j in range(i + 1, len(x_vals)):
                for k in range(len(y_vals)):
                    # k < l
                    for l in range(k + 1, len(y_vals)):
                        determinant = joint[i, k] * joint[j, l] - joint[j, k] * joint[i, l]
                        sum_sq_determinants += determinant ** 2

        marginal = np.sum(joint, axis=1)

        denominator = 0.0
        for i in range(len(x_vals)):
            # i < j
            for j in range(i + 1, len(x_vals)):
                denominator += marginal[i] ** 2 * marginal[j] ** 2

        return sum_sq_determinants / denominator


class RenyiCorrelation(_DependenceMeasure):
    """Renyi correlation. Measures how dependent two random variables are.

    As defined in this paper: https://link.springer.com/content/pdf/10.1007/BF02024507.pdf ,
    titled "On Measures of Dependence" by Alfréd Rényi.
    """

    _base_name: ClassVar[str] = "Renyi"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        base_values = actual.y if self.base == "y" else actual.s
        return self._corr(base_values.to_numpy().ravel(), prediction.hard.to_numpy().ravel())

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        x_vals = np.unique(x)
        y_vals = np.unique(y)
        if len(x_vals) < 2 or len(y_vals) < 2:
            return 1.0

        total = len(x)
        assert total == len(y)

        joint = np.empty((len(x_vals), len(y_vals)))

        for i, x_val in enumerate(x_vals):
            for k, y_val in enumerate(y_vals):
                # count how often x_val and y_val co-occur
                joint[i, k] = _count_true((x == x_val) & (y == y_val)) / total

        marginal_rows = np.sum(joint, axis=0, keepdims=True)
        marginal_cols = np.sum(joint, axis=1, keepdims=True)
        q_matrix = joint / np.sqrt(marginal_rows) / np.sqrt(marginal_cols)
        # singular value decomposition of Q
        singulars = np.linalg.svd(q_matrix, compute_uv=False)

        # return second-largest singular value
        return singulars[1]


def _count_true(mask: np.ndarray) -> int:
    """Count the number of elements that are True."""
    return mask.nonzero()[0].shape[0]
