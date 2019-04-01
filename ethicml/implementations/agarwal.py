"""Implementation of logistic regression (actually just a wrapper around sklearn)"""
import pandas as pd

from fairlearn.classred import expgrad
from fairlearn.moments import Moment, DP, EO
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ethicml.algorithms.utils import DataTuple
from ethicml.implementations.utils import instance_weight_check
from ethicml.utility.heaviside import Heaviside
from .common import InAlgoInterface


def train_and_predict(train: DataTuple, test: DataTuple,
                      classifier: str, fairness: str,
                      eps: float, iters: int):
    """Train a logistic regression model and compute predictions on the given test data"""

    train, _ = instance_weight_check(train)

    fairness_class: Moment = DP() if fairness == "DP" else EO()
    if classifier == "SVM":
        model = SVC(gamma='auto', random_state=888)
    else:
        model = LogisticRegression(solver='liblinear', random_state=888, max_iter=5000)

    data_x = train.x
    data_y = train.y[train.y.columns[0]]
    data_a = train.s[train.s.columns[0]]

    res_tuple = expgrad(dataX=data_x, dataA=data_a, dataY=data_y,
                        learner=model, cons=fairness_class, eps=eps, T=iters)

    res = res_tuple._asdict()

    preds = pd.DataFrame(res['best_classifier'](test.x), columns=["preds"])
    helper = Heaviside()
    preds = preds.apply(helper.apply)
    min_class_label = train.y[train.y.columns[0]].min()
    if preds['preds'].min() != preds['preds'].max():
        preds = preds.replace(preds['preds'].min(), min_class_label)
    return preds


def main():
    """This function runs the Agarwal model as a standalone program"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    classifier, fairness, eps, iters, = interface.remaining_args()
    interface.save_predictions(train_and_predict(train, test,
                                                 classifier=classifier, fairness=fairness,
                                                 eps=float(eps), iters=int(iters)))


if __name__ == "__main__":
    main()
