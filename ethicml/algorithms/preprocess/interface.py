"""Methods that define commandline interfaces"""
from typing import List, Any, Dict
from dataclasses import asdict

from ethicml.utility.data_structures import PathTuple, TestPathTuple


def flag_interface(
    train_paths: PathTuple,
    test_paths: TestPathTuple,
    new_train_paths: PathTuple,
    new_test_paths: TestPathTuple,
    flags: Dict[str, Any],
) -> List[str]:
    """Generate the commandline arguments that are expected"""
    flags_list: List[str] = []

    # paths to training and test data
    path_tuples = [train_paths, test_paths]
    prefixes = ["--train_", "--test_"]
    for path_tuple, prefix in zip(path_tuples, prefixes):
        for key, path in asdict(path_tuple).items():
            flags_list += [f"{prefix}{key}", str(path)]

    # paths to output files
    flags_list += [
        "--new_train_x",
        str(new_train_paths.x),
        "--new_train_s",
        str(new_train_paths.s),
        "--new_train_y",
        str(new_train_paths.y),
        "--new_train_name",
        new_train_paths.name,
        "--new_test_x",
        str(new_test_paths.x),
        "--new_test_s",
        str(new_test_paths.s),
        "--new_test_name",
        new_test_paths.name,
    ]
    print("flags_list", flags_list)

    # model parameters
    for key, values in flags.items():
        flags_list.append(f"--{key}")
        if isinstance(values, (list, tuple)):
            flags_list += [str(value) for value in values]
        else:
            flags_list.append(str(values))
    return flags_list
