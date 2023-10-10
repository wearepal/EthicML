"""Implementation of logistic regression (actually just a wrapper around sklearn)."""
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from ethicml.models.inprocess.in_subprocess import InAlgoArgs


def main() -> None:
    """This function runs the logistic regression model as a standalone program."""
    args: InAlgoArgs = json.loads(sys.argv[1])
    assert args["mode"] == "run", "model doesn’t support the fit/predict split yet"
    train, test = np.load(Path(args["train"])), np.load(Path(args["test"]))
    clf = LogisticRegression(solver="liblinear", random_state=888, multi_class="auto")
    clf.fit(train["x"], train["y"].ravel())
    predictions = clf.predict(test["x"])
    np.savez(Path(args["predictions"]), hard=predictions)


if __name__ == "__main__":
    main()
