"""Post-processing method by Hardt et al."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ethicml.utility.data_structures import DataTuple, TestTuple
from ethicml.metrics import TPR, TNR
from ethicml.evaluators import metric_per_sensitive_attribute
from .post_algorithm import PostAlgorithm


class Hardt(PostAlgorithm):
    """Post-processing method by Hardt et al."""

    def __init__(self):
        super().__init__()
        self._random = np.random.RandomState(seed=888)

    def run(
        self, train_predictions: pd.DataFrame, train: DataTuple, test_predictions, test: TestTuple
    ) -> pd.DataFrame:
        unfavorable_label = 0
        favorable_label = 1
        model_params = self._fit(train_predictions, train, favorable_label, unfavorable_label)
        return self._predict(
            model_params, test_predictions, test, favorable_label, unfavorable_label
        )

    def _fit(
        self,
        train_predictions: pd.DataFrame,
        train: DataTuple,
        favorable_label: int,
        unfavorable_label: int,
    ):
        # compute basic statistics
        sbr = metric.num_instances(privileged=True) / metric.num_instances()
        obr = metric.num_instances(privileged=False) / metric.num_instances()

        tprs = metric_per_sensitive_attribute(train_predictions, train, TPR())
        tpr0 = tprs['s_0']
        tpr1 = tprs['s_1']
        fnr0 = 1 - tpr0
        fnr1 = 1 - tpr1
        tnrs = metric_per_sensitive_attribute(train_predictions, train, TNR())
        tnr0 = tnrs['s_0']
        tnr1 = tnrs['s_1']
        fpr0 = 1 - tnr0
        fpr1 = 1 - tnr1

        # linear program has 4 decision variables:
        # [P[label_tilde = 1 | label_hat = 1, sensitive_attribute = 0];
        #  P[label_tilde = 1 | label_hat = 0, sensitive_attribute = 0];
        #  P[label_tilde = 1 | label_hat = 1, sensitive_attribute = 1];
        #  P[label_tilde = 1 | label_hat = 0, sensitive_attribute = 1]]
        # Coefficients of the linear objective function to be minimized.
        coeffs = np.array([fpr0 - tpr0, tnr0 - fnr0, fpr1 - tpr1, tnr1 - fnr1])

        # A_ub - 2-D array which, when matrix-multiplied by x, gives the values
        # of the upper-bound inequality constraints at x
        # b_ub - 1-D array of values representing the upper-bound of each
        # inequality constraint (row) in A_ub.
        # Just to keep these between zero and one
        A_ub = np.array(
            [
                [1, 0, 0, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, -1],
            ],
            dtype=np.float64,
        )
        b_ub = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)

        # Create boolean conditioning vectors for protected groups
        cond_vec_priv = compute_boolean_conditioning_vector(
            train.s, train.protected_attribute_names, self.privileged_groups
        )
        cond_vec_unpriv = compute_boolean_conditioning_vector(
            train.s, train.protected_attribute_names, self.unprivileged_groups
        )

        sconst = np.ravel(train_predictions[cond_vec_priv] == favorable_label)
        sflip = np.ravel(train_predictions[cond_vec_priv] == unfavorable_label)
        oconst = np.ravel(train_predictions[cond_vec_unpriv] == favorable_label)
        oflip = np.ravel(train_predictions[cond_vec_unpriv] == unfavorable_label)

        y_true = train.y.to_numpy().ravel()

        sm_tn = np.logical_and(sflip, y_true[cond_vec_priv] == unfavorable_label)
        sm_fn = np.logical_and(sflip, y_true[cond_vec_priv] == favorable_label)
        sm_fp = np.logical_and(sconst, y_true[cond_vec_priv] == unfavorable_label)
        sm_tp = np.logical_and(sconst, y_true[cond_vec_priv] == favorable_label)

        om_tn = np.logical_and(oflip, y_true[cond_vec_unpriv] == unfavorable_label)
        om_fn = np.logical_and(oflip, y_true[cond_vec_unpriv] == favorable_label)
        om_fp = np.logical_and(oconst, y_true[cond_vec_unpriv] == unfavorable_label)
        om_tp = np.logical_and(oconst, y_true[cond_vec_unpriv] == favorable_label)

        # A_eq - 2-D array which, when matrix-multiplied by x,
        # gives the values of the equality constraints at x
        # b_eq - 1-D array of values representing the RHS of each equality
        # constraint (row) in A_eq.
        # Used to impose equality of odds constraint
        A_eq = [
            [
                (np.mean(sconst * sm_tp) - np.mean(sflip * sm_tp)) / sbr,
                (np.mean(sflip * sm_fn) - np.mean(sconst * sm_fn)) / sbr,
                (np.mean(oflip * om_tp) - np.mean(oconst * om_tp)) / obr,
                (np.mean(oconst * om_fn) - np.mean(oflip * om_fn)) / obr,
            ],
            [
                (np.mean(sconst * sm_fp) - np.mean(sflip * sm_fp)) / (1 - sbr),
                (np.mean(sflip * sm_tn) - np.mean(sconst * sm_tn)) / (1 - sbr),
                (np.mean(oflip * om_fp) - np.mean(oconst * om_fp)) / (1 - obr),
                (np.mean(oconst * om_tn) - np.mean(oflip * om_tn)) / (1 - obr),
            ],
        ]

        b_eq = [
            (np.mean(oflip * om_tp) + np.mean(oconst * om_fn)) / obr
            - (np.mean(sflip * sm_tp) + np.mean(sconst * sm_fn)) / sbr,
            (np.mean(oflip * om_fp) + np.mean(oconst * om_tn)) / (1 - obr)
            - (np.mean(sflip * sm_fp) + np.mean(sconst * sm_tn)) / (1 - sbr),
        ]

        # Linear program
        return linprog(coeffs, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    def _predict(
        self,
        model_params,
        test_predictions: pd.DataFrame,
        test: TestTuple,
        favorable_label: int,
        unfavorable_label: int,
    ) -> pd.DataFrame:
        sp2p, sn2p, op2p, on2p = model_params.x

        # Create boolean conditioning vectors for protected groups
        cond_vec_priv = compute_boolean_conditioning_vector(
            test.s, test.protected_attribute_names, self.privileged_groups
        )
        cond_vec_unpriv = compute_boolean_conditioning_vector(
            test.s, test.protected_attribute_names, self.unprivileged_groups
        )

        # Randomly flip labels according to the probabilities in model_params
        self_fair_pred = test_predictions[cond_vec_priv].copy()
        self_pp_indices, _ = np.nonzero(test_predictions[cond_vec_priv] == favorable_label)
        self_pn_indices, _ = np.nonzero(test_predictions[cond_vec_priv] == unfavorable_label)
        self._random.shuffle(self_pp_indices)
        self._random.shuffle(self_pn_indices)

        n2p_indices = self_pn_indices[: int(len(self_pn_indices) * sn2p)]
        self_fair_pred[n2p_indices] = favorable_label
        p2n_indices = self_pp_indices[: int(len(self_pp_indices) * (1 - sp2p))]
        self_fair_pred[p2n_indices] = unfavorable_label

        othr_fair_pred = test_predictions[cond_vec_unpriv].copy()
        othr_pp_indices, _ = np.nonzero(test_predictions[cond_vec_unpriv] == favorable_label)
        othr_pn_indices, _ = np.nonzero(test_predictions[cond_vec_unpriv] == unfavorable_label)
        self._random.shuffle(othr_pp_indices)
        self._random.shuffle(othr_pn_indices)

        n2p_indices = othr_pn_indices[: int(len(othr_pn_indices) * on2p)]
        othr_fair_pred[n2p_indices] = favorable_label
        p2n_indices = othr_pp_indices[: int(len(othr_pp_indices) * (1 - op2p))]
        othr_fair_pred[p2n_indices] = unfavorable_label

        new_labels = np.zeros_like(test_predictions, dtype=np.float64)
        new_labels[cond_vec_priv] = self_fair_pred
        new_labels[cond_vec_unpriv] = othr_fair_pred

        return pd.DataFrame(new_labels, columns=['prediction'])

    @property
    def name(self) -> str:
        return "Hardt"
