"""Useful functions used in implementations."""
from pathlib import Path
from typing import Tuple

import tap

from ethicml.utility import DataTuple, TestTuple


class AlgoArgs(tap.Tap):
    """Base arguments needed for all algorithms."""

    # paths to the files with the data
    train: str
    test: str


class PreAlgoArgs(AlgoArgs):
    """ArgumentParser for pre-processing algorithms.

    This is a quick way to create a parser that can parse the filenames for the data. This class
    can be used by pre-algorithms to implement a commandline interface.
    """

    # paths to where the processed inputs should be stored
    new_train: str
    new_test: str


class InAlgoArgs(AlgoArgs):
    """ArgumentParser that already has arguments for the paths needed for InAlgorithms."""

    # path to where the predictions should be stored
    predictions: str


def load_data_from_flags(args: AlgoArgs) -> Tuple[DataTuple, TestTuple]:
    """Load data from the paths specified in the flags."""
    return DataTuple.from_npz(Path(args.train)), TestTuple.from_npz(Path(args.test))


def save_transformations(transforms: Tuple[DataTuple, TestTuple], args: PreAlgoArgs):
    """Save the data to the file that was specified in the commandline arguments."""
    train, test = transforms
    assert isinstance(train, DataTuple)
    assert isinstance(test, TestTuple)
    train.to_npz(Path(args.new_train))
    test.to_npz(Path(args.new_test))
