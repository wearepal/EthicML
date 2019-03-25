"""Implementation of SVM (actually just a wrapper around sklearn)"""
from sklearn.svm import SVC

import pandas as pd

from .common_in import InAlgoInterface


def train_and_predict(train, test, C, kernel):
    """Train an SVM model and compute predictions on the given test data"""
    clf = SVC(gamma='auto', random_state=888, C=C, kernel=kernel)
    clf.fit(train.x, train.y.values.ravel())
    return pd.DataFrame(clf.predict(test.x), columns=["preds"])


def main():
    """main method to run model"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    C, kernel = interface.remaining_args()
    interface.save_predictions(train_and_predict(train, test, C=float(C), kernel=kernel))


if __name__ == "__main__":
    main()
