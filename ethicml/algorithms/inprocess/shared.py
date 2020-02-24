"""Methods that are shared among the inprocess algorithms."""
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ethicml.utility import PathTuple, TestPathTuple

__all__ = ["flag_interface", "settings_for_svm_lr"]


def flag_interface(
    train_paths: PathTuple, test_paths: TestPathTuple, pred_path: Path, flags: Dict[str, Any]
) -> List[str]:
    """Generate the commandline arguments that are expected."""
    flags_list: List[str] = []

    # paths to training and test data
    path_tuples = [train_paths, test_paths]
    prefixes = ["--train_", "--test_"]
    for path_tuple, prefix in zip(path_tuples, prefixes):
        for key, path in asdict(path_tuple).items():
            flags_list += [f"{prefix}{key}", str(path)]

    # paths to output files
    flags_list += ["--pred_path", str(pred_path)]

    # model parameters
    for key, values in flags.items():
        flags_list.append(f"--{key}")
        if isinstance(values, (list, tuple)):
            flags_list += [str(value) for value in values]
        else:
            flags_list.append(str(values))
    return flags_list


def settings_for_svm_lr(
    classifier: str, C: Optional[float], kernel: Optional[str]
) -> Tuple[float, str]:
    """If necessary get the default settings for the C and kernel parameter of SVM and LR."""
    if C is None:
        if classifier == "LR":
            C = LogisticRegression().C
        elif classifier == "SVM":
            C = SVC().C
        else:
            raise NotImplementedError(f'Unsupported classifier "{classifier}".')

    if kernel is None:
        if classifier == "SVM":
            kernel = SVC().kernel
        else:
            kernel = ""

    return C, kernel
