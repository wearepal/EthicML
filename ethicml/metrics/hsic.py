"""Method for calculating the HSIC - an independence criterion.

a score of 0 denotes independence
"""
import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.random import RandomState

from ethicml.metrics.metric import MetricStaticName
from ethicml.utility import EvalTuple, Prediction

__all__ = ["Hsic"]


def hsic(
    prediction: np.ndarray, label: np.ndarray, sigma_first: float, sigma_second: float
) -> float:
    """Calculate the HSIC value.

    :param prediction:
    :param label:
    :param sigma_first:
    :param sigma_second:
    """
    xx_gram = np.array(np.matmul(np.expand_dims(prediction, 1), np.expand_dims(prediction, 1).T))
    yy_gram = np.array(np.matmul(np.expand_dims(label, 1), np.expand_dims(label, 1).T))

    x_sqnorms = np.diag(xx_gram)
    y_sqnorms = np.diag(yy_gram)

    def exp_r(x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x, 0)

    def exp_c(x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x, 1)

    gamma_first = 1.0 / (2 * sigma_first**2)
    gamma_second = 1.0 / (2 * sigma_second**2)
    # use the second binomial formula
    kernel_xx = np.exp(-gamma_first * (-2 * xx_gram + exp_c(x_sqnorms) + exp_r(x_sqnorms)))
    kernel_yy = np.exp(-gamma_second * (-2 * yy_gram + exp_c(y_sqnorms) + exp_r(y_sqnorms)))

    kernel_xx_mean = np.mean(kernel_xx)
    kernel_yy_mean = np.mean(kernel_yy)

    h_k = kernel_xx - kernel_xx_mean
    h_l = kernel_yy - kernel_yy_mean

    num = float(kernel_yy.shape[0])
    h_kf = h_k / (num - 1)
    h_lf = h_l / (num - 1)

    # biased estimate
    hsic_value = float(np.trace(np.matmul(h_kf.T, h_lf)))
    return hsic_value


@dataclass
class Hsic(MetricStaticName):
    """See module string."""

    seed: int = 888
    _name: ClassVar[str] = "HSIC"
    apply_per_sensitive: ClassVar[bool] = False

    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        """We add the ability to take the average of hsic score.

        As for larger datasets it will kill your machine

        :param prediction:
        :param actual:
        """
        preds = prediction.hard.to_numpy()[:, np.newaxis]
        sens_labels = actual.s.to_numpy()[:, np.newaxis]

        batchs_size = 5000

        together = np.hstack((preds, sens_labels)).transpose()

        random = RandomState(seed=self.seed)
        col_idx = random.permutation(together.shape[1])

        together = np.take(together, col_idx, axis=1)

        prediction_shuffled = together[0]
        label_shuffled = together[1]

        num_batches_float = preds.shape[0] / batchs_size
        num_batches: int = int(math.ceil(num_batches_float))

        batches = []

        start = 0

        for _ in range(num_batches):

            end = start + batchs_size

            preds_to_test = prediction_shuffled[start:end]
            labels_to_test = label_shuffled[start:end]

            batches.append(hsic(preds_to_test, labels_to_test, 0.7, 0.5))

            start += batchs_size

        return np.array(batches).mean().item()
