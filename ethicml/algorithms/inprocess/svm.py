"""
Wrapper for SKLearn implementation of SVM
"""

import pandas as pd

from sklearn.svm import SVC
from typing import List

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm


class SVM(InAlgorithm):
    """Support Vector Machine"""
    def _run(self, train, test):
        if self.hyperparams:
            if self.hyperparams.get('kernel') and self.hyperparams.get('C'):
                kernel = self.hyperparams['kernel']
                c = self.hyperparams['C']
                clf = SVC(C=c, kernel=kernel, gamma='auto', random_state=888)
                print("ok")
            elif self.hyperparams.get('kernel'):
                print("hmm")
            elif self.hyperparams.get('C'):
                c = list(self.hyperparams.values())[0]
                clf = SVC(C=c, gamma='auto', random_state=888)
        else:
            clf = SVC(gamma='auto', random_state=888)
        clf.fit(train.x, train.y.values.ravel())
        return pd.DataFrame(clf.predict(test.x), columns=["preds"])

    @property
    def name(self) -> str:
        return "SVM"

    @property
    def tunable_params(self) -> List[str]:
        return ['C']


def main():
    """main method to run model"""
    model = SVM()
    train, test = model.load_data()
    model.save_predictions(model.run(train, test))


if __name__ == "__main__":
    main()
