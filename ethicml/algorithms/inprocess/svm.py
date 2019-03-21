"""
Wrapper for SKLearn implementation of SVM
"""

import pandas as pd

from sklearn.svm import SVC
from typing import List

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm


def update_model(model, hyperparams):
    klass = model.__class__
    params = model.get_params()
    for hyperparam_name, hyperparam_new_val in hyperparams.items():
        params[hyperparam_name] = hyperparam_new_val
    return klass(**params)


class SVM(InAlgorithm):
    """Support Vector Machine"""
    def _run(self, train, test):
        clf = SVC(gamma='auto', random_state=888)
        clf = update_model(clf, self.hyperparams)
        clf.fit(train.x, train.y.values.ravel())
        return pd.DataFrame(clf.predict(test.x), columns=["preds"])

    @property
    def name(self) -> str:
        return "SVM"

    @property
    def tunable_params(self) -> List[str]:
        return ['C', 'kernel']


def main():
    """main method to run model"""
    model = SVM()
    train, test = model.load_data()
    model.save_predictions(model.run(train, test))


if __name__ == "__main__":
    main()
