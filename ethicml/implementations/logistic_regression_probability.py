"""
Logistic regaression with soft output
"""
from sklearn.linear_model import LogisticRegression

import pandas as pd

from ethicml.implementations.utils import instance_weight_check, InAlgoInterface


def train_and_predict(train, test, C):
    """Train a logistic regression model and compute predictions on the given test data"""
    clf = LogisticRegression(solver='liblinear', random_state=888, C=C)

    train, i_w = instance_weight_check(train)

    clf.fit(train.x, train.y.values.ravel(), sample_weight=i_w)
    return pd.DataFrame(clf.predict_proba(test.x)[:, 1], columns=["preds"])


def main():
    """main method to run model"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    C, = interface.remaining_args()
    interface.save_predictions(train_and_predict(train, test, float(C)))


if __name__ == "__main__":
    main()
