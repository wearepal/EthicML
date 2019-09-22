"""Implementation of logistic regression (actually just a wrapper around sklearn)"""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
import pandas as pd


def main():
    """This function runs the logistic regression model as a standalone program"""
    args = sys.argv[1:]
    x, y = pd.read_feather(Path(args[0])), pd.read_feather(Path(args[1]))
    test_x = pd.read_feather(Path(args[2]))
    clf = LogisticRegression(solver="liblinear", random_state=888, multi_class='auto')
    clf.fit(x, y.to_numpy().ravel())
    predictions = pd.DataFrame(clf.predict(test_x), columns=["preds"])
    predictions.to_feather(Path(args[3]))


if __name__ == "__main__":
    main()
