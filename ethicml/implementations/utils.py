"""Useful functions used in implementations."""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from ethicml.utility import DataTuple, SubgroupTuple, TestTuple

if TYPE_CHECKING:
    from ethicml.models.inprocess.in_subprocess import InAlgoRunArgs
    from ethicml.models.preprocess.pre_subprocess import PreAlgoRunArgs


def load_data_from_flags(args: PreAlgoRunArgs | InAlgoRunArgs) -> tuple[DataTuple, SubgroupTuple]:
    """Load data from the paths specified in the flags."""
    return DataTuple.from_file(Path(args["train"])), SubgroupTuple.from_file(Path(args["test"]))


def save_transformations(
    transforms: tuple[DataTuple, TestTuple], pre_algo_run_args: PreAlgoRunArgs
) -> None:
    """Save the data to the file that was specified in the commandline arguments."""
    train, test = transforms
    assert isinstance(train, DataTuple)
    assert isinstance(test, (DataTuple, SubgroupTuple))
    train.save_to_file(Path(pre_algo_run_args["new_train"]))
    test.save_to_file(Path(pre_algo_run_args["new_test"]))
