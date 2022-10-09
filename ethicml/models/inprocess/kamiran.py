"""Kamiran and Calders 2012."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from ranzen import implements
import sklearn
from sklearn.linear_model import LogisticRegression

from ethicml.models.inprocess.in_algorithm import InAlgorithm
from ethicml.models.inprocess.shared import settings_for_svm_lr
from ethicml.models.inprocess.svm import select_svm
from ethicml.utility import (
    ClassifierType,
    DataTuple,
    HyperParamType,
    KernelType,
    Prediction,
    SoftPrediction,
    TestTuple,
)

__all__ = ["Reweighting", "compute_instance_weights"]


VALID_MODELS = {ClassifierType.lr, ClassifierType.svm}


@dataclass
class Reweighting(InAlgorithm):
    """An implementation of the Reweighing method from `Kamiran and Calders 2012 <https://link.springer.com/article/10.1007/s10115-011-0463-8>`_.

    Each sample is assigned an instance-weight based on the joing probability of S and Y which is
    used during training of a classifier.

    :param classifier: The classifier to use.
    :param C: The C parameter for the classifier.
    :param kernel: The kernel to use for the classifier if SVM selected.
    """

    classifier: ClassifierType = ClassifierType.lr
    C: Optional[float] = None
    kernel: Optional[KernelType] = None

    def __post_init__(self) -> None:
        self.group_weights: Optional[Dict[str, Any]] = None
        self.chosen_c, self.chosen_kernel = settings_for_svm_lr(
            self.classifier, self.C, self.kernel
        )

    @property
    @implements(InAlgorithm)
    def hyperparameters(self) -> HyperParamType:
        _hyperparameters: dict[str, Any] = {"C": self.C}
        if self.classifier is ClassifierType.svm:
            assert self.kernel is not None
            _hyperparameters["kernel"] = self.kernel
        return _hyperparameters

    @property
    @implements(InAlgorithm)
    def name(self) -> str:
        lr_params = f" C={self.chosen_c}" if self.classifier is ClassifierType.lr else ""
        svm_params = (
            f" C={self.C}, kernel={self.chosen_kernel}"
            if self.classifier is ClassifierType.svm
            else ""
        )
        return f"Kamiran & Calders {self.classifier}{lr_params}{svm_params}"

    @implements(InAlgorithm)
    def fit(self, train: DataTuple, seed: int = 888) -> Reweighting:
        self.clf = self._train(
            train, classifier=self.classifier, C=self.chosen_c, kernel=self.chosen_kernel, seed=seed
        )
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return self._predict(model=self.clf, test=test)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        clf = self._train(
            train, classifier=self.classifier, C=self.chosen_c, kernel=self.chosen_kernel, seed=seed
        )
        return self._predict(model=clf, test=test)

    def _train(
        self,
        train: DataTuple,
        classifier: ClassifierType,
        C: float,
        kernel: KernelType | None,
        seed: int,
    ) -> sklearn.linear_model._base.LinearModel:
        if classifier is ClassifierType.svm:
            assert kernel is not None
            model = select_svm(C=C, kernel=kernel, seed=seed)
        else:
            random_state = np.random.RandomState(seed=seed)
            model = LogisticRegression(
                solver="liblinear", random_state=random_state, max_iter=50_00, C=C
            )
        weights = compute_instance_weights(train)["instance weights"]
        model.fit(
            train.x,
            train.y.to_numpy().ravel(),
            sample_weight=weights,
        )
        weights = weights.value_counts().rename_axis('weight').reset_index(name='count')
        groups = (
            pd.concat([train.s, train.y], axis=1)
            .groupby([train.s.name, train.y.name])
            .size()
            .reset_index(name="count")
        )
        self.group_weights = pd.merge(weights, groups, on="count").T.to_dict()
        return model

    def _predict(
        self, model: sklearn.linear_model._base.LinearModel, test: TestTuple
    ) -> Prediction:
        return SoftPrediction((model.predict_proba(test.x)), info=self.hyperparameters)


def compute_instance_weights(
    train: DataTuple, balance_groups: bool = False, upweight: bool = False
) -> pd.DataFrame:
    """Compute weights for all samples.

    :param train: The training data.
    :param balance_groups: Whether to balance the groups. When False, the groups are balanced as in
        `Kamiran and Calders 2012 <https://link.springer.com/article/10.1007/s10115-011-0463-8>`_.
        When True, the groups are numerically balanced. (Default: False)
    :param upweight: If balance_groups is True, whether to upweight the groups, or to downweight
        them. Downweighting is done by multiplying the weights by the inverse of the group size and
        is more numerically stable for small group sizes. (Default: False)
    :returns: A dataframe with the instance weights for each sample in the training data.
    """
    num_samples = len(train.x)
    s_unique, inv_indices_s, counts_s = np.unique(train.s, return_inverse=True, return_counts=True)
    _, inv_indices_y, counts_y = np.unique(train.y, return_inverse=True, return_counts=True)
    group_ids = (inv_indices_y * len(s_unique) + inv_indices_s).squeeze()
    gi_unique, inv_indices_gi, counts_joint = np.unique(
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

    return pd.DataFrame(group_weights[inv_indices_gi], columns=["instance weights"])
