"""
Wrapper for SKLearn implementation of SVM
"""
import pandas as pd
from sklearn.svm import SVC

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm


class SVM(InAlgorithm):
    """Support Vector Machine"""
    def __init__(self, C=None, kernel=None):
        super().__init__()
        self.C = C
        self.C = SVC().C if C is None else C
        self.kernel = SVC().kernel if kernel is None else kernel

    def _run(self, train, test):
        clf = SVC(gamma='auto', random_state=888, C=self.C, kernel=self.kernel)
        clf.fit(train.x, train.y.values.ravel())
        return pd.DataFrame(clf.predict(test.x), columns=["preds"])

    @property
    def name(self) -> str:
        return "SVM"


def main():
    """main method to run model"""
    model = SVM()
    train, test = model.load_data()
    model.save_predictions(model.run(train, test))


if __name__ == "__main__":
    main()
