"""Implementation of SVM (actually just a wrapper around sklearn)"""
from sklearn.svm import SVC, LinearSVC

import pandas as pd

from ethicml.implementations.utils import instance_weight_check, InAlgoInterface


def train_and_predict(train, test, C, kernel):
    """Train an SVM model and compute predictions on the given test data"""
    if kernel == 'linear':
        clf = LinearSVC(tol=1e-12, dual=False, random_state=888, C=C)
    else:
        clf = SVC(gamma='auto', random_state=888, C=C, kernel=kernel)
    train, i_w = instance_weight_check(train)
    clf.fit(train.x, train.y.values.ravel(), sample_weight=i_w)
    return pd.DataFrame(clf.predict(test.x), columns=["preds"])


def main():
    """This function runs the SVM model as a standalone program"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    C, kernel = interface.remaining_args()
    interface.save_predictions(train_and_predict(train, test, C=float(C), kernel=kernel))


if __name__ == "__main__":
    main()
