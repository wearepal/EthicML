"""Useful functions used in implementations."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from ethicml.utility import DataTuple, SubgroupTuple, TestTuple

if TYPE_CHECKING:
    from ethicml.algorithms.inprocess.in_subprocess import InAlgoRunArgs
    from ethicml.algorithms.preprocess.pre_subprocess import PreAlgoRunArgs


def load_data_from_flags(args: PreAlgoRunArgs | InAlgoRunArgs) -> Tuple[DataTuple, SubgroupTuple]:
    """Load data from the paths specified in the flags.

    :param args:
    """
    return DataTuple.from_npz(Path(args["train"])), SubgroupTuple.from_npz(Path(args["test"]))


def save_transformations(
    transforms: Tuple[DataTuple, TestTuple], pre_algo_run_args: PreAlgoRunArgs
) -> None:
    """Save the data to the file that was specified in the commandline arguments.

    :param transforms:
    :param pre_algo_run_args:
    """
    train, test = transforms
    assert isinstance(train, DataTuple)
    assert isinstance(test, (DataTuple, SubgroupTuple))
    train.to_npz(Path(pre_algo_run_args["new_train"]))
    test.to_npz(Path(pre_algo_run_args["new_test"]))
