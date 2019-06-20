"""
Returns a subset of the data. Used primarily in testing so that kernel methods finish in a
reasonable time
"""
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass, astuple
from enum import Enum

import pandas as pd


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class TestTuple:
    """A tuple of dataframes for the features and the sensitive attribute"""

    x: pd.DataFrame  # features
    s: pd.DataFrame  # senstitive attributes


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class DataTuple(TestTuple):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels"""

    y: pd.DataFrame  # class labels


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class TestPathTuple:
    """For algorithms that run in their own process, we pass around paths to the data"""

    x: Path  # path to file with features
    s: Path  # path to file with sensitive attributes


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class PathTuple(TestPathTuple):
    """For algorithms that run in their own process, we pass around paths to the data"""

    y: Path  # path to file with class labels


def write_as_feather(
    train: DataTuple, test: TestTuple, data_dir: Path
) -> (Tuple[PathTuple, TestPathTuple]):
    """Write the given DataTuple to Feather files and return the file paths as PathTuples

    Args:
        train: tuple with training data
        test: tuple with test data
        data_dir: directory where the files should be stored
    Returns:
        tuple of tuple of paths (one tuple for training, one for test)
    """
    # create the directory if it doesn't already exist
    data_dir.mkdir(parents=True, exist_ok=True)

    def _save(data: pd.DataFrame, prefix: str, key: str) -> Path:
        # SUGGESTION: maybe the file names should be completely random to avoid collisions
        data_path = data_dir / f"data_{prefix}_{key}.feather"
        # write the file
        data.to_feather(data_path)
        return data_path

    train_paths = PathTuple(
        x=_save(train.x, "train", "x"),
        s=_save(train.s, "train", "s"),
        y=_save(train.y, "train", "y"),
    )
    test_paths = TestPathTuple(x=_save(test.x, "test", "x"), s=_save(test.s, "test", "s"))
    return train_paths, test_paths


def apply_to_joined_tuple(mapper, datatup: DataTuple) -> DataTuple:
    """Concatenate the dataframes in a DataTuple and apply a function to it"""
    cols_x = datatup.x.columns
    cols_s = datatup.s.columns
    cols_y = datatup.y.columns
    joined = pd.concat([datatup.x, datatup.s, datatup.y], axis="columns", sort=False)
    joined = mapper(joined)
    return DataTuple(x=joined[cols_x], s=joined[cols_s], y=joined[cols_y])


def concat_dt(datatup_list: List[DataTuple], axis: str = "index", ignore_index: bool = False):
    """Concatenate the data tuples in the given list"""
    return DataTuple(
        *[
            pd.concat(
                [astuple(datatup)[i] for datatup in datatup_list],
                axis=axis,
                sort=False,
                ignore_index=ignore_index,
            )
            for i, _ in enumerate(astuple(datatup_list[0]))
        ]
    )


def load_feather(output_path: Path) -> pd.DataFrame:
    """Load a dataframe from a feather file"""
    with output_path.open("rb") as file_obj:
        df = pd.read_feather(file_obj)
    return df


def get_subset(train: DataTuple, num: int = 500) -> DataTuple:
    """Get the first elements of the given dataset

    Args:
        train: training data

    Returns:
        subset of training data
    """
    return DataTuple(x=train.x.iloc[:num], s=train.s.iloc[:num], y=train.y.iloc[:num])


class FairType(Enum):
    """
    This is an enum that can be used to specify the type of fairness

    It basically works like the enums in C, but we can also give it values that are not integers.
    """
    DI = "DI"
    DP = "DI"  # alias for DI (for "demographic parity")
    EOPP = "Eq. Opp"
    EODDS = "Eq. Odds"

    def __str__(self) -> str:
        """This function is needed so that, for example, str(FairType.DI) returns 'DI'."""
        return self.value
