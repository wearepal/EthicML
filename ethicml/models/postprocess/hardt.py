"""Post-processing method by Hardt et al."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.random import RandomState
import pandas as pd
from ranzen import implements
from scipy.optimize import OptimizeResult, linprog  # type: ignore[attr-defined]

from ethicml.metrics.per_sensitive_attribute import metric_per_sens
from ethicml.metrics.tnr import TNR
from ethicml.metrics.tpr import TPR
from ethicml.utility import DataTuple, Prediction, TestTuple

from .post_algorithm import PostAlgorithm

__all__ = ["Hardt"]


@dataclass
class Hardt(PostAlgorithm):
    """Post-processing method by Hardt et al."""

    unfavorable_label: int = 0
    favorable_label: int = 1

    @property
    @implements(PostAlgorithm)
    def name(self) -> str:
        return "Hardt"

    @implements(PostAlgorithm)
    def fit(self, train_predictions: Prediction, train: DataTuple) -> Hardt:
        self.model_params = self._fit(train_predictions, train)
        return self

    @implements(PostAlgorithm)
    def predict(self, test_predictions: Prediction, test: TestTuple, seed: int = 888) -> Prediction:
        return self._predict(self.model_params, test_predictions, test, seed)

    @implements(PostAlgorithm)
    def run(
        self,
        train_predictions: Prediction,
        train: DataTuple,
        test_predictions: Prediction,
        test: TestTuple,
        seed: int = 888,
    ) -> Prediction:
        model_params = self._fit(train_predictions, train)
        return self._predict(model_params, test_predictions, test, seed)

    def _fit(self, train_predictions: Prediction, train: DataTuple) -> OptimizeResult:
        # compute basic statistics
        fraction_s0 = (train.s.to_numpy() == 0).mean()
        fraction_s1 = 1 - fraction_s0

        s_col = train.s.name
        tprs = metric_per_sens(train_predictions, train, TPR())
        tpr0 = tprs[f"{s_col}_0"]
        tpr1 = tprs[f"{s_col}_1"]
        fnr0 = 1 - tpr0
        fnr1 = 1 - tpr1
        tnrs = metric_per_sens(train_predictions, train, TNR())
        tnr0 = tnrs[f"{s_col}_0"]
        tnr1 = tnrs[f"{s_col}_1"]
        fpr0 = 1 - tnr0
        fpr1 = 1 - tnr1

        # linear program has 4 decision variables:
        # [P[label_tilde = 1 | label_hat = 1, sensitive_attribute = 0];
        #  P[label_tilde = 1 | label_hat = 0, sensitive_attribute = 0];
        #  P[label_tilde = 1 | label_hat = 1, sensitive_attribute = 1];
        #  P[label_tilde = 1 | label_hat = 0, sensitive_attribute = 1]]
        # Coefficients of the linear objective function to be minimized.
        coeffs = np.array([fpr0 - tpr0, tnr0 - fnr0, fpr1 - tpr1, tnr1 - fnr1])

        # inequalilty_constraint_matrix: 2-D array which, when matrix-multiplied by x, gives the
        # values of the upper-bound inequality constraints at x
        # b_ub: 1-D array of values representing the upper-bound of each
        # inequality constraint (row) in A_ub.
        # Just to keep these between zero and one
        inequalilty_constraint_matrix: np.ndarray = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        b_ub: np.ndarray = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)

        # Create boolean conditioning vectors for protected groups
        mask_s1 = train.s.to_numpy() == 1
        mask_s0 = train.s.to_numpy() == 0

        train_preds_numpy: np.ndarray = train_predictions.hard.to_numpy()

        sconst = np.ravel(train_preds_numpy[mask_s1] == self.favorable_label)
        sflip = np.ravel(train_preds_numpy[mask_s1] == self.unfavorable_label)
        oconst = np.ravel(train_preds_numpy[mask_s0] == self.favorable_label)
        oflip = np.ravel(train_preds_numpy[mask_s0] == self.unfavorable_label)

        y_true = train.y.to_numpy().ravel()

        sm_tn = np.logical_and(sflip, y_true[mask_s1] == self.unfavorable_label)
        sm_fn = np.logical_and(sflip, y_true[mask_s1] == self.favorable_label)
        sm_fp = np.logical_and(sconst, y_true[mask_s1] == self.unfavorable_label)
        sm_tp = np.logical_and(sconst, y_true[mask_s1] == self.favorable_label)

        om_tn = np.logical_and(oflip, y_true[mask_s0] == self.unfavorable_label)
        om_fn = np.logical_and(oflip, y_true[mask_s0] == self.favorable_label)
        om_fp = np.logical_and(oconst, y_true[mask_s0] == self.unfavorable_label)
        om_tp = np.logical_and(oconst, y_true[mask_s0] == self.favorable_label)

        # A_eq - 2-D array which, when matrix-multiplied by x,
        # gives the values of the equality constraints at x.
        # b_eq - 1-D array of values representing the RHS of each equality constraint (row) in A_eq.
        # Used to impose equality of odds constraint
        a_eq = [
            [
                (np.mean(sconst * sm_tp) - np.mean(sflip * sm_tp)) / fraction_s1,
                (np.mean(sflip * sm_fn) - np.mean(sconst * sm_fn)) / fraction_s1,
                (np.mean(oflip * om_tp) - np.mean(oconst * om_tp)) / fraction_s0,
                (np.mean(oconst * om_fn) - np.mean(oflip * om_fn)) / fraction_s0,
            ],
            [
                (np.mean(sconst * sm_fp) - np.mean(sflip * sm_fp)) / (1 - fraction_s1),
                (np.mean(sflip * sm_tn) - np.mean(sconst * sm_tn)) / (1 - fraction_s1),
                (np.mean(oflip * om_fp) - np.mean(oconst * om_fp)) / (1 - fraction_s0),
                (np.mean(oconst * om_tn) - np.mean(oflip * om_tn)) / (1 - fraction_s0),
            ],
        ]

        b_eq = [
            (np.mean(oflip * om_tp) + np.mean(oconst * om_fn)) / fraction_s0
            - (np.mean(sflip * sm_tp) + np.mean(sconst * sm_fn)) / fraction_s1,
            (np.mean(oflip * om_fp) + np.mean(oconst * om_tn)) / (1 - fraction_s0)
            - (np.mean(sflip * sm_fp) + np.mean(sconst * sm_tn)) / (1 - fraction_s1),
        ]

        # Linear program
        return linprog(coeffs, A_ub=inequalilty_constraint_matrix, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq)

    def _predict(
        self, model_params: OptimizeResult, test_predictions: Prediction, test: TestTuple, seed: int
    ) -> Prediction:
        random = RandomState(seed=seed)
        sp2p, sn2p, op2p, on2p = model_params.x

        # Create boolean conditioning vectors for protected groups
        mask_s1 = test.s.to_numpy() == 1
        mask_s0 = test.s.to_numpy() == 0

        test_preds_numpy: np.ndarray = test_predictions.hard.to_numpy()

        # Randomly flip labels according to the probabilities in model_params
        self_fair_pred = test_preds_numpy[mask_s1].copy()
        self_pp_indices = (test_preds_numpy[mask_s1] == self.favorable_label).nonzero()[0]
        self_pn_indices = (test_preds_numpy[mask_s1] == self.unfavorable_label).nonzero()[0]
        random.shuffle(self_pp_indices)
        random.shuffle(self_pn_indices)

        n2p_indices = self_pn_indices[: int(len(self_pn_indices) * sn2p)]
        self_fair_pred[n2p_indices] = self.favorable_label
        p2n_indices = self_pp_indices[: int(len(self_pp_indices) * (1 - sp2p))]
        self_fair_pred[p2n_indices] = self.unfavorable_label

        othr_fair_pred = test_preds_numpy[mask_s0].copy()
        othr_pp_indices = (test_preds_numpy[mask_s0] == self.favorable_label).nonzero()[0]
        othr_pn_indices = (test_preds_numpy[mask_s0] == self.unfavorable_label).nonzero()[0]
        random.shuffle(othr_pp_indices)
        random.shuffle(othr_pn_indices)

        n2p_indices = othr_pn_indices[: int(len(othr_pn_indices) * on2p)]
        othr_fair_pred[n2p_indices] = self.favorable_label
        p2n_indices = othr_pp_indices[: int(len(othr_pp_indices) * (1 - op2p))]
        othr_fair_pred[p2n_indices] = self.unfavorable_label

        new_labels = np.zeros_like(test_preds_numpy, dtype=np.float64)
        new_labels[mask_s1] = self_fair_pred
        new_labels[mask_s0] = othr_fair_pred

        return Prediction(hard=pd.Series(new_labels))
