"""Methods that are shared among the inprocess algorithms"""
from typing import List, Tuple, Optional, Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ethicml.utility.data_structures import PathTuple, TestPathTuple


def conventional_interface(
    train_paths: PathTuple, test_paths: TestPathTuple, *args: Any
) -> List[str]:
    """
    Generate the commandline arguments that are expected by the scripts that follow the convention.

    The agreed upon order is:
    x (train), s (train), y (train), x (test), s (test), predictions.
    """
    list_to_return = [
        str(train_paths.x),
        str(train_paths.s),
        str(train_paths.y),
        str(train_paths.name),
        str(test_paths.x),
        str(test_paths.s),
        str(test_paths.name),
    ]
    for arg in args:
        list_to_return.append(str(arg))
    return list_to_return


def settings_for_svm_lr(
    classifier: str, C: Optional[float], kernel: Optional[str]
) -> Tuple[float, str]:
    """if necessary get the default settings for the C and kernel parameter of SVM and LR"""
    if C is None:
        if classifier == "LR":
            C = LogisticRegression().C
        elif classifier == "SVM":
            C = SVC().C
        else:
            raise NotImplementedError(f"Unsupported classifier \"{classifier}\".")

    if kernel is None:
        if classifier == "SVM":
            kernel = SVC().kernel
        else:
            kernel = ""

    return C, kernel
