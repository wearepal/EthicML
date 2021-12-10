"""Kamiran and Calders 2012."""
from typing import Optional

import numpy as np
import pandas as pd
from ranzen import implements
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
        seed: int = 888,
    ):
        super().__init__(name=f"Kamiran & Calders {classifier}")
        if classifier not in VALID_MODELS:
            raise ValueError(f"results: classifier must be one of {VALID_MODELS!r}.")
        self.classifier = classifier
        self.C, self.kernel = settings_for_svm_lr(classifier, C, kernel)
        self.seed = seed

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        return _train_and_predict(
            train, test, classifier=self.classifier, C=self.C, kernel=self.kernel, seed=self.seed
        )


def compute_instance_weights(
    train: DataTuple, balance_groups: bool = False, upweight: bool = False
) -> pd.DataFrame:
    """Compute weights for all samples."""
    num_samples = len(train.x)
    s_unique, inv_indexes_s, counts_s = np.unique(train.s, return_inverse=True, return_counts=True)
    _, inv_indexes_y, counts_y = np.unique(train.y, return_inverse=True, return_counts=True)
    group_ids = (inv_indexes_y * len(s_unique) + inv_indexes_s).squeeze()
    gi_unique, inv_indexes_gi, counts_joint = np.unique(
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
        counts_factorized = np.outer(counts_y, counts_s).flatten()
        group_weights = counts_factorized[gi_unique] / (num_samples * counts_joint)

    return pd.DataFrame(group_weights[inv_indexes_gi], columns=["instance weights"])


def _train_and_predict(
    train: DataTuple, test: TestTuple, classifier: ClassifierType, C: float, kernel: str, seed: int
) -> Prediction:
    """Train a logistic regression model and compute predictions on the given test data."""
    if classifier == "SVM":
        model = select_svm(C=C, kernel=kernel, seed=seed)
    else:
        random_state = np.random.RandomState(seed=seed)
        model = LogisticRegression(
            solver="liblinear", random_state=random_state, max_iter=5000, C=C
        )
    model.fit(
        train.x,
        train.y.to_numpy().ravel(),
        sample_weight=compute_instance_weights(train)["instance weights"],
    )
    return Prediction(hard=pd.Series(model.predict(test.x)))
