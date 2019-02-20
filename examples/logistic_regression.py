"""
Logistic regression that doesn't import anything from ethicml (i.e. it's *standalone*)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


def main(args):
    """train a logistic regression model and save the predictions in a numpy file"""
    # load data
    xtrain = load_dataframe(Path(args[1]))
    ytrain = load_dataframe(Path(args[2]))
    xtest = load_dataframe(Path(args[3]))

    # path where the results should be stored
    pred_path = Path(args[4])

    clf = LogisticRegression(random_state=888)
    clf.fit(xtrain, ytrain.values.ravel())
    predictions = clf.predict(xtest)
    np.save(pred_path, predictions)


def load_dataframe(path):
    """Load dataframe from a parquet file"""
    with path.open('rb') as f:
        df = pd.read_parquet(f)
    return df


if __name__ == "__main__":
    main(sys.argv)
