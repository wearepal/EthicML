"""Kamiran and Calders 2012."""
from __future__ import annotations

from typing import Optional

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


def _get_groups_ids(dt: DataTuple) -> np.ndarray:
    """Compute the unique group id for each sample based on its s and y labels."""
    return (dt.y.to_numpy() * len(np.unique(dt.s)) + dt.s).squeeze()


def compute_instance_weights(train: DataTuple, upweight: bool = False) -> pd.DataFrame:
    """Compute weights for all samples."""
    group_ids = _get_groups_ids(train)
    _, indexes, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
    # Upweight samples according to the cardinality of their intersectional group
    if upweight:
        group_weights = len(group_ids) / counts
    # Downweight samples according to the cardinality of their intersectional group
    # - this approach should be preferred due to being more numerically stable
    # (very small counts can lead to very large weighted loss values when upweighting)
    else:
        group_weights = 1 - (counts / len(group_ids))
    return pd.DataFrame(group_weights[indexes], columns=["instance weights"])


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
