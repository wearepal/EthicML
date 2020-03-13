"""Methods that define commandline interfaces."""
from pathlib import Path
from typing import Any, Dict, List


def flag_interface(
    train_path: Path,
    test_path: Path,
    new_train_path: Path,
    new_test_path: Path,
    flags: Dict[str, Any],
) -> List[str]:
    """Generate the commandline arguments that are expected by the script about to be called."""
    # paths to training and test data
    data_flags: Dict[str, Any] = {"train": train_path, "test": test_path}
    # paths to output files
    data_flags.update({"new_train": new_train_path, "new_test": new_test_path})
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
