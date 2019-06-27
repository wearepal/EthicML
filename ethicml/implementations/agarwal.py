"""Implementation of logistic regression (actually just a wrapper around sklearn)"""
import pandas as pd

from fairlearn.classred import expgrad
from fairlearn.moments import Moment, DP, EO
from sklearn.linear_model import LogisticRegression

from ethicml.utility.data_structures import DataTuple, TestTuple
from ethicml.implementations.utils import InAlgoInterface
from ethicml.implementations.svm import select_svm
from ethicml.utility.heaviside import Heaviside


def train_and_predict(
    train: DataTuple,
    test: TestTuple,
    classifier: str,
    fairness: str,
    eps: float,
    iters: int,
    C: float,
    kernel: str,
):
    """Train a logistic regression model and compute predictions on the given test data"""

    fairness_class: Moment = DP() if fairness == "DP" else EO()
    if classifier == "SVM":
        model = select_svm(C, kernel)
    else:
        model = LogisticRegression(solver="liblinear", random_state=888, max_iter=5000, C=float(C))

    data_x = train.x
    data_y = train.y[train.y.columns[0]]
    data_a = train.s[train.s.columns[0]]

    res_tuple = expgrad(
        dataX=data_x,
        dataA=data_a,
        dataY=data_y,
        learner=model,
        cons=fairness_class,
        eps=eps,
        T=iters,
    )

    res = res_tuple._asdict()

    preds = pd.DataFrame(res["best_classifier"](test.x), columns=["preds"])
    helper = Heaviside()
    preds = preds.apply(helper.apply)
    min_class_label = train.y[train.y.columns[0]].min()
    if preds["preds"].min() != preds["preds"].max():
        preds = preds.replace(preds["preds"].min(), min_class_label)
    return preds


def main():
    """This function runs the Agarwal model as a standalone program"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    classifier, fairness, eps, iters, C, kernel = interface.remaining_args()
    interface.save_predictions(
        train_and_predict(
            train,
            test,
            classifier=classifier,
            fairness=fairness,
            eps=float(eps),
            iters=int(iters),
            C=float(C),
            kernel=kernel,
        )
    )


if __name__ == "__main__":
    main()
