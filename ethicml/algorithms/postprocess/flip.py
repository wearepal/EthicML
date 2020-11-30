"""Demographic Parity Label flipping approach."""
from abc import abstractmethod
from typing import Tuple

import numpy as np

from ethicml import DataTuple, PostAlgorithm, Prediction, TestTuple, implements


class FlipBase(PostAlgorithm):
    """Common for flipping functions."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    @implements(PostAlgorithm)
    def run(
        self,
        train_predictions: Prediction,
        train: DataTuple,
        test_predictions: Prediction,
        test: TestTuple,
    ) -> Prediction:
        x, y = self._fit(test, test_predictions)
        _test_preds = self._flip(test_predictions, test, flip_0_to_1=True, num_to_flip=x, s_group=0)
        return self._flip(_test_preds, test, flip_0_to_1=False, num_to_flip=y, s_group=1)

    @abstractmethod
    def _fit(self, test: TestTuple, preds: Prediction) -> Tuple[int, int]:
        pass


class DPFlip(FlipBase):
    """Randomly flip a number of decisions such that perfect demographic parity is achieved."""

    def __init__(self) -> None:
        super().__init__(name="DemPar. Post Process")

    def _flip(
        self,
        preds: Prediction,
        dt: TestTuple,
        flip_0_to_1: bool,
        num_to_flip: int,
        s_group: int,
    ) -> Prediction:
        if num_to_flip >= 0:
            pre_y_val = 0 if flip_0_to_1 else 1
            post_y_val = 1 if flip_0_to_1 else 0
        else:
            pre_y_val = 1 if flip_0_to_1 else 0
            post_y_val = 0 if flip_0_to_1 else 1
            num_to_flip = abs(num_to_flip)

        _y = preds.hard[preds.hard == pre_y_val]
        _s = preds.hard[dt.s[dt.s.columns[0]] == s_group]
        idx_s_y = _y.index & _s.index  # type: ignore[operator]
        rng = np.random.RandomState(888)
        idxs = [i for i in rng.permutation(idx_s_y)]
        preds.hard.update({idx: post_y_val for idx in idxs[:num_to_flip]})
        return preds

    def _fit(self, test: TestTuple, preds: Prediction) -> Tuple[int, int]:
        y_0 = preds.hard[preds.hard == 0]
        y_1 = preds.hard[preds.hard == 1]
        s_0 = test.s[test.s[test.s.columns[0]] == 0]
        s_1 = test.s[test.s[test.s.columns[0]] == 1]
        # Naming is nSY
        n00 = preds.hard[(s_0.index) & (y_0.index)].count()  # type: ignore[operator]
        n01 = preds.hard[(s_0.index) & (y_1.index)].count()  # type: ignore[operator]
        n10 = preds.hard[(s_1.index) & (y_0.index)].count()  # type: ignore[operator]
        n11 = preds.hard[(s_1.index) & (y_1.index)].count()  # type: ignore[operator]

        a = (((n00 + n01) * n11) - ((n10 + n11) * n01)) / (n00 + n01)
        b = (n10 + n11) / (n00 + n01)

        if b > 1:
            x = a / b
            z = 0
        else:
            x = 0
            z = a
        return int(round(x)), int(round(z))


class EqOppFlip(FlipBase):
    """Randomly flip a number of decisions such that perfect TPR parity is achieved.

    This Shouldn't be used. Use Hardt et al instead. This method requires access to the ground
    truth labels, which is very bad.
    """

    def __init__(self) -> None:
        super().__init__(name="EqOpp. Post Process")

    def _flip(
        self,
        preds: Prediction,
        dt: TestTuple,
        flip_0_to_1: bool,
        num_to_flip: int,
        s_group: int,
    ) -> Prediction:
        if num_to_flip >= 0:
            pre_y_val = 0 if flip_0_to_1 else 1
            post_y_val = 1 if flip_0_to_1 else 0
        else:
            pre_y_val = 1 if flip_0_to_1 else 0
            post_y_val = 0 if flip_0_to_1 else 1
            num_to_flip = abs(num_to_flip)

        _pred = preds.hard[preds.hard == pre_y_val]
        _s = preds.hard[dt.s[dt.s.columns[0]] == s_group]
        _y = preds.hard[dt.y[dt.y.columns[0]] == 1]
        idx_s_y = _pred.index & _s.index & _y.index  # type: ignore[operator]
        if len(idx_s_y) < num_to_flip:
            raise AssertionError("Not enough valid candidates to flip")

        rng = np.random.RandomState(888)
        idxs = [i for i in rng.permutation(idx_s_y)]
        preds.hard.update({idx: post_y_val for idx in idxs[:num_to_flip]})
        return preds

    def _fit(self, test: DataTuple, preds: Prediction) -> Tuple[int, int]:
        preds_0 = preds.hard[preds.hard == 0]
        preds_1 = preds.hard[preds.hard == 1]
        s_0 = test.s[test.s[test.s.columns[0]] == 0]
        s_1 = test.s[test.s[test.s.columns[0]] == 1]
        y_0 = test.y[test.y[test.y.columns[0]] == 0]
        y_1 = test.y[test.y[test.y.columns[0]] == 1]

        r0 = preds.hard[(s_0.index) & (y_1.index) & (preds_1.index)].count()  # type: ignore[operator]
        r1 = preds.hard[(s_1.index) & (y_1.index) & (preds_1.index)].count()  # type: ignore[operator]

        # Naming is nSY
        n00 = preds.hard[(s_0.index) & (y_0.index)].count()  # type: ignore[operator]
        n01 = preds.hard[(s_0.index) & (y_1.index)].count()  # type: ignore[operator]
        n10 = preds.hard[(s_1.index) & (y_0.index)].count()  # type: ignore[operator]
        n11 = preds.hard[(s_1.index) & (y_1.index)].count()  # type: ignore[operator]

        a = r1 - ((n11 * r0) / n01)
        b = n11 / n01

        if b > 1:
            x = a / b
            z = 0
        else:
            x = 0
            z = a
        return int(round(x)), int(round(z))
