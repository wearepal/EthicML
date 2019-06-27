"""Methods that define commandline interfaces"""
from typing import List

from ethicml.utility.data_structures import PathTuple, TestPathTuple


def conventional_interface(train_paths: PathTuple, test_paths: TestPathTuple, *args) -> List[str]:
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
