"""Implementation of logistic regression (actually just a wrapper around sklearn)"""
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

import pandas as pd

from ethicml.implementations.utils import InAlgoInterface


def train_and_predict(train, test):
    """Train a logistic regression model and compute predictions on the given test data"""
    folder = KFold(n_splits=3, random_state=888, shuffle=False)
    clf = LogisticRegressionCV(
        cv=folder, n_jobs=-1, random_state=888, solver="liblinear", multi_class='auto'
    )
    clf.fit(train.x, train.y.values.ravel())
    return pd.DataFrame(clf.predict(test.x), columns=["preds"])


def main():
    """This function runs the cross-validated logistic regression model as a standalone program"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    interface.save_predictions(train_and_predict(train, test))


if __name__ == "__main__":
    main()
