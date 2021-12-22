"""Methods that are shared among the inprocess algorithms."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

__all__ = ["flag_interface", "settings_for_svm_lr"]


def flag_interface(
    flags: Dict[str, Any],
    *,
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    pred_path: Optional[Path] = None,
) -> List[str]:
    """Generate the commandline arguments that are expected by the script about to be called."""
    # paths to training and test data
    data_flags: Dict[str, Any] = {}
    if train_path is not None:
        data_flags["train"] = train_path
    if test_path is not None:
        data_flags["test"] = test_path
    if model_path is not None:
        data_flags["model"] = model_path
    if pred_path is not None:
        data_flags["predictions"] = pred_path
    data_flags.update(flags)

    flags_list: List[str] = []
    # model parameters
    for key, values in data_flags.items():
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
        kernel = SVC().kernel if classifier == "SVM" else ""
    return C, kernel
