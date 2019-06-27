"""Implementation of SVM (actually just a wrapper around sklearn)"""
from sklearn.svm import SVC, LinearSVC

import pandas as pd

from ethicml.utility.data_structures import DataTuple, TestTuple
from ethicml.implementations.utils import InAlgoInterface


def select_svm(C: float, kernel: str):
    """Select the appropriate SVM model for the given parameters"""
    if kernel == "linear":
        return LinearSVC(C=C, dual=False, tol=1e-12, random_state=888)
    return SVC(C=C, kernel=kernel, gamma="auto", random_state=888)


def train_and_predict(train: DataTuple, test: TestTuple, C: float, kernel: str) -> pd.DataFrame:
    """Train an SVM model and compute predictions on the given test data"""
    clf = select_svm(C, kernel)
    clf.fit(train.x, train.y.to_numpy().ravel())
    return pd.DataFrame(clf.predict(test.x), columns=["preds"])


def main():
    """This function runs the SVM model as a standalone program"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    C, kernel = interface.remaining_args()
    interface.save_predictions(train_and_predict(train, test, C=float(C), kernel=kernel))


if __name__ == "__main__":
    main()
