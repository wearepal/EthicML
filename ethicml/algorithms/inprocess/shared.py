"""Methods that are shared among the inprocess algorithms."""
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ethicml.utility.data_structures import PathTuple, TestPathTuple


def flag_interface(
    train_path: PathTuple, test_path: TestPathTuple, pred_path: Path, flags: Dict[str, Any]
) -> List[str]:
    """Generate the commandline arguments that are expected."""
    # paths to training and test data
    data_flags = {"train_data_path": train_path.data_path, "train_name": train_path.name}
    data_flags.update({"test_data_path": test_path.data_path, "test_name": test_path.name})
    data_flags.update({"pred_path": pred_path})
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
        if classifier == "SVM":
            kernel = SVC().kernel
        else:
            kernel = ""

    return C, kernel
