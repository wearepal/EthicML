"""Methods that define commandline interfaces."""
from pathlib import Path
from typing import Any, Dict, List, Optional


def flag_interface(
    flags: Dict[str, Any],
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    new_train_path: Optional[Path] = None,
    new_test_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> List[str]:
    """Generate the commandline arguments that are expected by the script about to be called."""
    # paths to training and test data
    data_flags: Dict[str, Any] = {}
    if train_path is not None:
        data_flags["train"] = str(train_path)
    if new_train_path is not None:
        data_flags["new_train"] = str(new_train_path)
    if test_path is not None:
        data_flags["test"] = str(test_path)
    if new_test_path is not None:
        data_flags["new_test"] = str(new_test_path)
    if model_path is not None:
        data_flags["model"] = str(model_path)
    # paths to output files
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
