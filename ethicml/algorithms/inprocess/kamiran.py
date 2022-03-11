"""Kamiran and Calders 2012."""
from typing import Optional

import numpy as np
import pandas as pd
import sklearn.linear_model._base
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
        self.seed = seed
        if classifier not in VALID_MODELS:
            raise ValueError(f"results: classifier must be one of {VALID_MODELS!r}.")
        self.classifier = classifier
        self.C, self.kernel = settings_for_svm_lr(classifier, C, kernel)
        self.is_fairness_algo = True

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return f"Kamiran & Calders {self.classifier}"

    @implements(InAlgorithm)
    def fit(self, train: DataTuple) -> InAlgorithm:
        self.clf = _train(
            train, classifier=self.classifier, C=self.C, kernel=self.kernel, seed=self.seed
        )
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return _predict(model=self.clf, test=test)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        clf = _train(
            train, classifier=self.classifier, C=self.C, kernel=self.kernel, seed=self.seed
        )
        return _predict(model=clf, test=test)


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


def _train(
    train: DataTuple, classifier: ClassifierType, C: float, kernel: str, seed: int
) -> sklearn.linear_model._base.LinearModel:
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
    return model


def _predict(model: sklearn.linear_model._base.LinearModel, test: TestTuple) -> Prediction:
    return Prediction(hard=pd.Series(model.predict(test.x)))
