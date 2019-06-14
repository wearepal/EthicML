"""Implementation of logistic regression (actually just a wrapper around sklearn)"""
from sklearn.linear_model import LogisticRegression

import pandas as pd

from ethicml.implementations.utils import InAlgoInterface


def train_and_predict(train, test, C):
    """Train a logistic regression model and compute predictions on the given test data"""
    clf = LogisticRegression(solver="liblinear", random_state=888, C=C, multi_class='auto')
    clf.fit(train.x, train.y.values.ravel())
    return pd.DataFrame(clf.predict(test.x), columns=["preds"])


def main():
    """This function runs the logistic regression model as a standalone program"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    C, = interface.remaining_args()
    interface.save_predictions(train_and_predict(train, test, C=float(C)))


if __name__ == "__main__":
    main()
