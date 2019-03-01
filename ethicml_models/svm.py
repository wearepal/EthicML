"""
SVM that doesn't import anything from EthicML (i.e. it's *standalone*)
"""
from sklearn.svm import SVC

from .common import CommonIn


def main():
    """train an SVM model and save the predictions in a numpy file"""
    # parse args and load data
    algo_in = CommonIn()
    train, test = algo_in.load_data()

    clf = SVC(random_state=888)
    clf.fit(train.x, train.y.values.ravel())
    algo_in.save_predictions(clf.predict(test.x))


if __name__ == "__main__":
    main()
