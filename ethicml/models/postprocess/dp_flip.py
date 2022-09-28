"""Demographic Parity Label flipping approach."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ranzen import implements

from ethicml.utility import DataTuple, Prediction, TestTuple

from .post_algorithm import PostAlgorithm

__all__ = ["DPFlip"]


@dataclass
class DPFlip(PostAlgorithm):
    """Randomly flip a number of decisions such that perfect demographic parity is achieved."""

    @property
    @implements(PostAlgorithm)
    def name(self) -> str:
        return "DemPar. Post Process"

    @implements(PostAlgorithm)
    def fit(self, train_predictions: Prediction, train: DataTuple) -> DPFlip:
        return self

    @implements(PostAlgorithm)
    def predict(self, test_predictions: Prediction, test: TestTuple, seed: int = 888) -> Prediction:
        x, y = self._fit(test, test_predictions)
        _test_preds = self._flip(
            test_predictions, test, flip_0_to_1=True, num_to_flip=x, s_group=0, seed=seed
        )
        return self._flip(_test_preds, test, flip_0_to_1=False, num_to_flip=y, s_group=1, seed=seed)

    @implements(PostAlgorithm)
    def run(
        self,
        train_predictions: Prediction,
        train: DataTuple,
        test_predictions: Prediction,
        test: TestTuple,
        seed: int = 888,
    ) -> Prediction:
        x, y = self._fit(test, test_predictions)
        _test_preds = self._flip(
            test_predictions, test, flip_0_to_1=True, num_to_flip=x, s_group=0, seed=seed
        )
        return self._flip(_test_preds, test, flip_0_to_1=False, num_to_flip=y, s_group=1, seed=seed)

    @staticmethod
    def _flip(
        preds: Prediction,
        dt: TestTuple,
        flip_0_to_1: bool,
        num_to_flip: int,
        s_group: int,
        seed: int,
    ) -> Prediction:
        if num_to_flip >= 0:
            pre_y_val = 0 if flip_0_to_1 else 1
            post_y_val = 1 if flip_0_to_1 else 0
        else:
            pre_y_val = 1 if flip_0_to_1 else 0
            post_y_val = 0 if flip_0_to_1 else 1
            num_to_flip = abs(num_to_flip)

        _y = preds.hard[preds.hard == pre_y_val]
        _s = preds.hard[dt.s == s_group]
        idx_s_y = _y.index.intersection(_s.index)
        rng = np.random.RandomState(seed)
        idxs = list(rng.permutation(idx_s_y))
        update = pd.Series({idx: post_y_val for idx in idxs[:num_to_flip]}, dtype=preds.hard.dtype)
        preds.hard.update(update)
        return preds

    @staticmethod
    def _fit(test: TestTuple, preds: Prediction) -> tuple[int, int]:
        y_0 = preds.hard[preds.hard == 0]
        y_1 = preds.hard[preds.hard == 1]
        s_0 = test.s[test.s == 0]
        s_1 = test.s[test.s == 1]
        # Naming is nSY
        n00 = preds.hard[s_0.index.intersection(y_0.index)].count()
        n01 = preds.hard[s_0.index.intersection(y_1.index)].count()
        n10 = preds.hard[s_1.index.intersection(y_0.index)].count()
        n11 = preds.hard[s_1.index.intersection(y_1.index)].count()

        a = (((n00 + n01) * n11) - ((n10 + n11) * n01)) / (n00 + n01)
        b = (n10 + n11) / (n00 + n01)

        if b > 1:
            x = a / b
            z = 0.0
        else:
            x = 0.0
            z = a
        return int(round(x)), int(round(z))
