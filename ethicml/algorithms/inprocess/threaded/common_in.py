"""Threaded logistic regression"""
from .threaded_in_algorithm import BasicTIA


class CommonTIA(BasicTIA):
    """
    Class that works with all algorithm scripts that follow a certain convention with regards to
    their commandline interface
    """
    @staticmethod
    def _script_interface(train_paths, test_paths, pred_path):
        """
        Generate the commandline arguments that are expected by the scripts that follow the
        convention.

        The agreed upon order is:
        x (train), s (train), y (train), x (test), s (test), y (test), predictions.
        """
        return [
            str(train_paths.x),
            str(train_paths.s),
            str(train_paths.y),
            str(test_paths.x),
            str(test_paths.s),
            str(test_paths.y),
            str(pred_path)
        ]


class ThreadedLR(CommonTIA):
    """Threaded logistic regression"""
    def __init__(self):
        super().__init__(name="threaded_LR", script_path="-m ethicml_models.logistic_regression")


class ThreadedSVM(CommonTIA):
    """Threaded SVM"""
    def __init__(self):
        super().__init__(name="threaded_SVM", script_path="-m ethicml_models.svm")
