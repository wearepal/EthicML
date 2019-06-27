"""
Implementatioon of Kamiran and Calders 2012

Heavily based on AIF360
https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/reweighing.py

    References:
        .. [4] F. Kamiran and T. Calders,  "Data Preprocessing Techniques for
           Classification without Discrimination," Knowledge and Information
           Systems, 2012.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from ethicml.utility.data_structures import DataTuple
from ethicml.implementations.utils import InAlgoInterface
from ethicml.implementations.svm import select_svm


def _obtain_conditionings(dataset: DataTuple):
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


def compute_weights(train: DataTuple) -> pd.DataFrame:
    """Compute weights for all samples"""
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

    w_p_fav = n_fav * n_p / (num_samples * n_p_fav)
    w_p_unfav = n_unfav * n_p / (num_samples * n_p_unfav)
    w_up_fav = n_fav * n_up / (num_samples * n_up_fav)
    w_up_unfav = n_unfav * n_up / (num_samples * n_up_unfav)

    train_instance_weights = pd.DataFrame(np.ones(train.x.shape[0]), columns=["instance weights"])

    train_instance_weights.iloc[cond_p_fav.index] *= w_p_fav
    train_instance_weights.iloc[cond_p_unfav.index] *= w_p_unfav
    train_instance_weights.iloc[cond_up_fav.index] *= w_up_fav
    train_instance_weights.iloc[cond_up_unfav.index] *= w_up_unfav

    return train_instance_weights


def train_and_predict(train, test, classifier, C: float, kernel: str):
    """Train a logistic regression model and compute predictions on the given test data"""
    if classifier == "SVM":
        model = select_svm(C, kernel)
    else:
        model = LogisticRegression(solver="liblinear", random_state=888, max_iter=5000, C=C)
    model.fit(
        train.x, train.y.values.ravel(), sample_weight=compute_weights(train)["instance weights"]
    )
    return pd.DataFrame(model.predict(test.x), columns=["preds"])


def main():
    """This function runs the Kamiran&Calders method as a standalone program"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    classifier, C, kernel = interface.remaining_args()
    interface.save_predictions(train_and_predict(train, test, classifier, float(C), kernel))


if __name__ == "__main__":
    main()
