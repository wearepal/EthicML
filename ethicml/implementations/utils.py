"""Useful functions used in implementations."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from ethicml.utility import DataTuple, TestTuple

if TYPE_CHECKING:
    from ethicml.algorithms.algorithm_base import AlgoArgs
    from ethicml.algorithms.preprocess.pre_algorithm import PreAlgoArgs


def load_data_from_flags(args: AlgoArgs) -> Tuple[DataTuple, TestTuple]:
    """Load data from the paths specified in the flags."""
    assert "train" in args
    assert "test" in args
    return DataTuple.from_npz(Path(args["train"])), TestTuple.from_npz(Path(args["test"]))


def save_transformations(transforms: Tuple[DataTuple, TestTuple], args: PreAlgoArgs) -> None:
    """Save the data to the file that was specified in the commandline arguments."""
    train, test = transforms
    assert isinstance(train, DataTuple)
    assert isinstance(test, TestTuple)
    assert "new_train" in args
    assert "new_test" in args
    train.to_npz(Path(args["new_train"]))
    test.to_npz(Path(args["new_test"]))
