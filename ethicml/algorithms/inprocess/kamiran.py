"""Kamiran and Calders 2012."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from kit import implements
from sklearn.linear_model import LogisticRegression

from ethicml.utility import ClassifierType, DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm
from .shared import settings_for_svm_lr
from .svm import select_svm

__all__ = ["Kamiran", "compute_instance_weights"]


VALID_MODELS = {"LR", "SVM"}


class Kamiran(InAlgorithm):
    """Kamiran and Calders 2012."""

    def __init__(
        self,
        classifier: ClassifierType = "LR",
        C: Optional[float] = None,
        kernel: Optional[str] = None,
    ):
        super().__init__(name=f"Kamiran & Calders {classifier}")
        if classifier not in VALID_MODELS:
            raise ValueError(f"results: classifier must be one of {VALID_MODELS!r}.")
        self.classifier = classifier
        self.C, self.kernel = settings_for_svm_lr(classifier, C, kernel)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        return _train_and_predict(
            train, test, classifier=self.classifier, C=self.C, kernel=self.kernel
        )


def _obtain_conditionings(
    dataset: DataTuple,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Obtain the necessary conditioning boolean vectors to compute instance level weights."""
    y_col = dataset.y.columns[0]
    y_pos = dataset.y[y_col].max()
    y_neg = dataset.y[y_col].min()
    s_col = dataset.s.columns[0]
    s_pos = dataset.s[s_col].max()
    s_neg = dataset.s[s_col].min()

    # combination of label and privileged/unpriv. groups
    cond_p_fav = dataset.x.loc[(dataset.y[y_col] == y_pos) & (dataset.s[s_col] == s_pos)]
    cond_p_unfav = dataset.x.loc[(dataset.y[y_col] == y_neg) & (dataset.s[s_col] == s_pos)]
    cond_up_fav = dataset.x.loc[(dataset.y[y_col] == y_pos) & (dataset.s[s_col] == s_neg)]
    cond_up_unfav = dataset.x.loc[(dataset.y[y_col] == y_neg) & (dataset.s[s_col] == s_neg)]

    return cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav


def compute_instance_weights(
    train: DataTuple, balance_groups: bool = False, upweight: bool = False
) -> pd.DataFrame:
    """Compute weights for all samples."""
    num_samples = len(train.x)
    s_unique, s_unique_counts = np.unique(train.s, return_counts=True)
    group_ids = (train.y.to_numpy() * len(s_unique) + train.s).squeeze()

    unique_ids, inv_indexes, counts_joint = np.unique(
        group_ids, return_inverse=True, return_counts=True
    )
    if balance_groups:
        # Upweight samples according to the cardinality of their intersectional group
        if upweight:
            group_weights = num_samples / counts_joint
        # Downweight samples according to the cardinality of their intersectional group
        # - this approach should be preferred due to being more numerically stable
        # (very small counts can lead to very large weighted loss values when upweighting)
        else:
            group_weights = 1 - (counts_joint / num_samples)
    else:
        _, y_unique_counts = np.unique(train.y, return_counts=True)
        counts_factorized = np.outer(y_unique_counts, s_unique_counts).flatten()
        group_weights = counts_factorized[unique_ids] / (num_samples * counts_joint)

    return pd.DataFrame(group_weights[inv_indexes], columns=["instance weights"])


def _train_and_predict(
    train: DataTuple, test: TestTuple, classifier: ClassifierType, C: float, kernel: str
) -> Prediction:
    """Train a logistic regression model and compute predictions on the given test data."""
    if classifier == "SVM":
        model = select_svm(C, kernel)
    else:
        model = LogisticRegression(solver="liblinear", random_state=888, max_iter=5000, C=C)
    model.fit(
        train.x,
        train.y.to_numpy().ravel(),
        sample_weight=compute_instance_weights(train)["instance weights"],
    )
    return Prediction(hard=pd.Series(model.predict(test.x)))
