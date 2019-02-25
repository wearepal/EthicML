"""Threaded logistic regression"""
from ethicml.common import ROOT_PATH
from .threaded_in_algorithm import SimpleTIA

EXAMPLES_PATH = ROOT_PATH.parent / "examples/"


class CommonInterface(SimpleTIA):
    """Class that works with all algorithm scripts that have a certain commandline interface"""
    @staticmethod
    def script_interface(train_paths, test_paths, pred_path):
        """Generate the commandline arguments that are expected by many of our scripts.

        The agreed upon order is: x (train), s (train), y (train), x (test), s (test), predictions.
        """
        return [
            str(train_paths.x),
            str(train_paths.s),
            str(train_paths.y),
            str(test_paths.x),
            str(test_paths.s),
            str(pred_path)
        ]


class ThreadedLR(CommonInterface):
    """Threaded logistic regression"""
    def __init__(self):
        super().__init__("threaded_LR", EXAMPLES_PATH / "logistic_regression.py")


class ThreadedSVM(CommonInterface):
    """Threaded SVM"""
    def __init__(self):
        super().__init__("threaded_SVM", EXAMPLES_PATH / "svm.py")
