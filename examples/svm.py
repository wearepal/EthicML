"""
SVM that doesn't import anything from EthicML (i.e. it's *standalone*)
"""
from sklearn.svm import SVC

from common import CommonInterface


def main():
    """train an SVM model and save the predictions in a numpy file"""
    # parse args and load data
    common_interface = CommonInterface()
    train, test = common_interface.load()

    clf = SVC(random_state=888)
    clf.fit(train.x, train.y.values.ravel())
    predictions = clf.predict(test.x)
    common_interface.save(predictions)


if __name__ == "__main__":
    main()
