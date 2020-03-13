"""Implementation of SVM (actually just a wrapper around sklearn)."""
from pathlib import Path

import numpy as np
from sklearn.svm import SVC, LinearSVC

from .utils import InAlgoArgs


class SvmArgs(InAlgoArgs):
    """Commandline arguments for SVM."""

    c: float
    kernel: str


def main():
    """This function runs the SVM model as a standalone program."""
    args = SvmArgs().parse_args()
    with open(args.train, "rb") as train_file:
        train = np.load(train_file)
        train_x, train_y = train["x"], train["y"]
    with open(args.test, "rb") as test_file:
        test = np.load(test_file)
        test_x = test["x"]
    if args.kernel == "linear":
        clf = LinearSVC(C=args.c, dual=False, tol=1e-12, random_state=888)
    else:
        clf = SVC(C=args.c, kernel=args.kernel, gamma="auto", random_state=888)
    clf.fit(train_x, train_y.ravel())
    predictions = clf.predict(test_x)
    np.savez(Path(args.predictions), hard=predictions)


if __name__ == "__main__":
    main()
