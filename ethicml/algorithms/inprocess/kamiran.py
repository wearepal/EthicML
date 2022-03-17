"""Kamiran and Calders 2012."""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import sklearn.linear_model._base
from ranzen import implements
from sklearn.linear_model import LogisticRegression

from ethicml.utility import ClassifierType, DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithm
from .shared import settings_for_svm_lr
from .svm import KernelType, select_svm

__all__ = ["Kamiran", "compute_instance_weights"]


VALID_MODELS = {"LR", "SVM"}


class Kamiran(InAlgorithm):
    """An implementation of the Reweighing method from `Kamiran and Calders 2012 <https://link.springer.com/article/10.1007/s10115-011-0463-8>`_.

    Each sample is assigned an instance-weight based on the joing probability of S and Y which is used during training of a classifier.
    """

    def __init__(
        self,
        *,
        classifier: ClassifierType = "LR",
        C: Optional[float] = None,
        kernel: Optional[KernelType] = None,
        seed: int = 888,
    ):
        """Reweighing.

        Args:
            classifier: The classifier to use.
            C: The C parameter for the classifier.
            kernel: The kernel to use for the classifier if SVM selected.
            seed: The random number generator seed to use for the classifier.
        """
        self.seed = seed
        if classifier not in VALID_MODELS:
            raise ValueError(f"results: classifier must be one of {VALID_MODELS!r}.")
        self.classifier = classifier
        self.C, self.kernel = settings_for_svm_lr(classifier, C, kernel)
        self.is_fairness_algo = True
        self._hyperparameters = {"C": self.C}
        if self.classifier == "SVM":
            self._hyperparameters["kernel"] = self.kernel
        self.group_weights: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        lr_params = f" C={self.C}" if self.classifier == "LR" else ""
        svm_params = f" C={self.C}, kernel={self.kernel}" if self.classifier == "SVM" else ""
        return f"Kamiran & Calders {self.classifier}{lr_params}{svm_params}"

    @implements(InAlgorithm)
    def fit(self, train: DataTuple) -> InAlgorithm:
        self.clf = self._train(
            train, classifier=self.classifier, C=self.C, kernel=self.kernel, seed=self.seed
        )
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        return self._predict(model=self.clf, test=test)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        clf = self._train(
            train, classifier=self.classifier, C=self.C, kernel=self.kernel, seed=self.seed
        )
        return self._predict(model=clf, test=test)

    def _train(
        self, train: DataTuple, classifier: ClassifierType, C: float, kernel: str, seed: int
    ) -> sklearn.linear_model._base.LinearModel:
        if classifier == "SVM":
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
        weights = weights.value_counts().rename_axis('weight').reset_index(name='count')  # type: ignore[union-attr]
        groups = (
            pd.concat([train.s, train.y], axis=1)
            .groupby([train.s.columns[0], train.y.columns[0]])
            .size()
            .reset_index(name="count")
        )
        self.group_weights = pd.merge(weights, groups, on="count").T.to_dict()
        return model

    def _predict(
        self, model: sklearn.linear_model._base.LinearModel, test: TestTuple
    ) -> Prediction:
        return Prediction(hard=pd.Series(model.predict(test.x)))


def compute_instance_weights(
    train: DataTuple, balance_groups: bool = False, upweight: bool = False
) -> pd.DataFrame:
    """Compute weights for all samples.

    Args:
        train: The training data.
        balance_groups: Whether to balance the groups. When False, the groups are balanced as in
            `Kamiran and Calders 2012 <https://link.springer.com/article/10.1007/s10115-011-0463-8>`_.
            When True, the groups are numerically balanced.
        upweight: If balance_groups is True, whether to upweight the groups, or to downweight them.
            Downweighting is done by multiplying the weights by the inverse of the group size
            and is more numerically stable for small group sizes.

    Returns:
        A dataframe with the instance weights for each sample in the training data.
    """
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
