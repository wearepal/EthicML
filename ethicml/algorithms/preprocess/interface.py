"""Methods that define commandline interfaces"""
from pathlib import Path
from typing import List, Any, Dict
from dataclasses import asdict

from ..utils import PathTuple, TestPathTuple


def flag_interface(
    train_paths: PathTuple,
    test_paths: TestPathTuple,
    new_train_path: Path,
    new_test_path: Path,
    flags: Dict[str, Any],
) -> List[str]:
    """Generate the commandline arguments that are expected"""
    flags_list: List[str] = []

    # paths to training and test data
    path_tuples = [train_paths, test_paths]
    prefixes = ['--train_', '--test_']
    for path_tuple, prefix in zip(path_tuples, prefixes):
        for key, path in asdict(path_tuple).items():
            flags_list += [f"{prefix}{key}", str(path)]

    # paths to output files
    flags_list += ['--train_new', str(new_train_path), '--test_new', str(new_test_path)]

    # model parameters
    for key, values in flags.items():
        flags_list.append(f"--{key}")
        if isinstance(values, list):
            flags_list += [str(value) for value in values]
        else:
            flags_list.append(str(values))
    return flags_list
