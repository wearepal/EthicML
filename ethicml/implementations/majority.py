"""Implementation of majority classifier"""

import pandas as pd

from ethicml.implementations.utils import InAlgoInterface


def train_and_predict(train, test):
    """return the majority label of the train set"""
    maj = train.y.mode().values
    return pd.DataFrame(maj.repeat(len(test.x)), columns=["preds"])


def main():
    """This function runs the majoirty model as a standalone program"""
    interface = InAlgoInterface()
    train, test = interface.load_data()
    interface.save_predictions(train_and_predict(train, test))


if __name__ == "__main__":
    main()
