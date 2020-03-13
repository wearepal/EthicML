"""Implementation of logistic regression (actually just a wrapper around sklearn)"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def main():
    """This function runs the logistic regression model as a standalone program"""
    args = sys.argv[1:]
    train, test = np.load(Path(args[0])), np.load(Path(args[1]))
    clf = LogisticRegression(solver="liblinear", random_state=888, multi_class="auto")
    clf.fit(train["x"], train["y"].ravel())
    predictions = clf.predict(test["x"])
    np.savez(Path(args[2]), hard=predictions)


if __name__ == "__main__":
    main()
