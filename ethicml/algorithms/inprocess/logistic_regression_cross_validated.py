"""
Implementation of cross validated LR. This is a work around for now,
long term we'll have a proper cross-validation mechanism
"""

import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.utils import DataTuple


class LRCV(InAlgorithm):

    """Kind of a cheap hack for now, but gives a proper cross-valudeted LR"""
    def run(self, train: DataTuple, test: DataTuple, sub_process: bool = False) -> pd.DataFrame:
        if sub_process:
            return self.run_threaded(train, test)

        folder = KFold(n_splits=3, random_state=888, shuffle=False)
        clf = LogisticRegressionCV(cv=folder, n_jobs=-1, random_state=888, solver='liblinear')
        clf.fit(train.x, train.y.values.ravel())
        return pd.DataFrame(clf.predict(test.x), columns=["preds"])

    @property
    def name(self) -> str:
        return "LRCV"


def main():
    """main method to run model"""
    model = LRCV()
    train, test = model.load_data()
    model.save_predictions(model.run(train, test))


if __name__ == "__main__":
    main()
