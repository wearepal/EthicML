"""Kamiran and Calders 2012."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ethicml.common import implements
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


def compute_instance_weights(train: DataTuple) -> pd.DataFrame:
    """Compute weights for all samples."""
    np.random.seed(888)
    (cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav) = _obtain_conditionings(train)

    y_col = train.y.columns[0]
    y_pos = train.y[y_col].max()
    y_neg = train.y[y_col].min()
    s_col = train.s.columns[0]
    s_pos = train.s[s_col].max()
    s_neg = train.s[s_col].min()

    num_samples = train.x.shape[0]
    n_p = train.s.loc[train.s[s_col] == s_pos].shape[0]
    n_up = train.s.loc[train.s[s_col] == s_neg].shape[0]
    n_fav = train.y.loc[train.y[y_col] == y_pos].shape[0]
    n_unfav = train.y.loc[train.y[y_col] == y_neg].shape[0]

    n_p_fav = cond_p_fav.shape[0]
    n_p_unfav = cond_p_unfav.shape[0]
    n_up_fav = cond_up_fav.shape[0]
    n_up_unfav = cond_up_unfav.shape[0]

    w_p_fav = n_fav * n_p / (num_samples * n_p_fav) if n_p_fav != 0 else float("NaN")
    w_p_unfav = n_unfav * n_p / (num_samples * n_p_unfav) if n_p_unfav != 0 else float("NaN")
    w_up_fav = n_fav * n_up / (num_samples * n_up_fav) if n_up_fav != 0 else float("NaN")
    w_up_unfav = n_unfav * n_up / (num_samples * n_up_unfav) if n_up_unfav != 0 else float("NaN")

    train_instance_weights = pd.DataFrame(np.ones(train.x.shape[0]), columns=["instance weights"])

    train_instance_weights.iloc[cond_p_fav.index] *= w_p_fav
    train_instance_weights.iloc[cond_p_unfav.index] *= w_p_unfav
    train_instance_weights.iloc[cond_up_fav.index] *= w_up_fav
    train_instance_weights.iloc[cond_up_unfav.index] *= w_up_unfav

    return train_instance_weights


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
