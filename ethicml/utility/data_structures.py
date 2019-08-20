"""
Returns a subset of the data. Used primarily in testing so that kernel methods finish in a
reasonable time
"""
from pathlib import Path
from typing import Tuple, List, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class TestTuple:
    """A tuple of dataframes for the features and the sensitive attribute"""

    x: pd.DataFrame  # features
    s: pd.DataFrame  # senstitive attributes
    name: Optional[str] = None  # name of the dataset

    def __iter__(self):
        return iter([self.x, self.s])

    def write_as_feather(self, data_dir: Path, identifier: str) -> "TestPathTuple":
        """Write the TestTuple to Feather files and return the file paths as a TestPathTuple

        Args:
            data_dir: directory where the files should be stored
            identifier: a string that ideally identifies the file uniquely, can be a UUID
        Returns:
            tuple of paths
        """
        # create the directory if it doesn't already exist
        data_dir.mkdir(parents=True, exist_ok=True)

        return TestPathTuple(
            x=_save_helper(data_dir, self.x, identifier, "x"),
            s=_save_helper(data_dir, self.s, identifier, "s"),
            name=self.name if self.name is not None else "",
        )


@dataclass(frozen=True)
class DataTupleValues:
    y: pd.DataFrame  # class labels


@dataclass(frozen=True)
class DataTuple(TestTuple, DataTupleValues):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels"""

    def __iter__(self):
        return iter([self.x, self.s, self.y])

    def remove_y(self) -> TestTuple:
        """Convert the DataTuple instance to a TestTuple instance"""
        return TestTuple(x=self.x, s=self.s, name=self.name)

    def write_as_feather(self, data_dir: Path, identifier: str) -> "PathTuple":
        """Write the DataTuple to Feather files and return the file paths as a PathTuple

        Args:
            data_dir: directory where the files should be stored
            identifier: a string that ideally identifies the file uniquely, can be a UUID
        Returns:
            tuple of paths
        """
        # create the directory if it doesn't already exist
        data_dir.mkdir(parents=True, exist_ok=True)

        return PathTuple(
            x=_save_helper(data_dir, self.x, identifier, "x"),
            s=_save_helper(data_dir, self.s, identifier, "s"),
            y=_save_helper(data_dir, self.y, identifier, "y"),
            name=self.name if self.name is not None else "",
        )


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class TestPathTuple:
    """For algorithms that run in their own process, we pass around paths to the data"""

    x: Path  # path to file with features
    s: Path  # path to file with sensitive attributes
    name: str  # name of the dataset

    def load_from_feather(self) -> TestTuple:
        """Load a dataframe from a feather file"""
        return TestTuple(
            x=load_feather(self.x), s=load_feather(self.s), name=self.name if self.name else None
        )


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class PathTuple(TestPathTuple):
    """For algorithms that run in their own process, we pass around paths to the data"""

    y: Path  # path to file with class labels

    def load_from_feather(self) -> DataTuple:
        """Load a dataframe from a feather file"""
        return DataTuple(
            x=load_feather(self.x),
            s=load_feather(self.s),
            y=load_feather(self.y),
            name=self.name if self.name else None,
        )


def _save_helper(data_dir: Path, data: pd.DataFrame, prefix: str, key: str) -> Path:
    # SUGGESTION: maybe the file names should be completely random to avoid collisions
    data_path = data_dir / f"data_{prefix}_{key}.feather"
    # write the file
    data.to_feather(data_path)
    return data_path


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
    # TODO: this should be an assert instead (requires changing code that calls this function)
    if isinstance(test, DataTuple):
        # because of polymorphism it can happen that `test` is a DataTuple posing as a TestTuple
        # this causes problems though because it will write an additional file (the one with y)
        test = test.remove_y()
    return train.write_as_feather(data_dir, "train"), test.write_as_feather(data_dir, "test")


def apply_to_joined_tuple(mapper, datatup: DataTuple) -> DataTuple:
    """Concatenate the dataframes in a DataTuple and apply a function to it"""
    cols_x = datatup.x.columns
    cols_s = datatup.s.columns
    cols_y = datatup.y.columns
    joined = pd.concat([datatup.x, datatup.s, datatup.y], axis="columns", sort=False)
    joined = mapper(joined)
    return DataTuple(x=joined[cols_x], s=joined[cols_s], y=joined[cols_y], name=datatup.name)


def concat_dt(
    datatup_list: List[DataTuple], axis: str = "index", ignore_index: bool = False
) -> DataTuple:
    """Concatenate the data tuples in the given list"""

    to_return = DataTuple(
        x=datatup_list[0].x, s=datatup_list[0].s, y=datatup_list[0].y, name=datatup_list[0].name
    )
    for i in range(1, len(datatup_list)):
        to_return = DataTuple(
            x=pd.concat(
                [to_return.x, datatup_list[i].x], axis=axis, sort=False, ignore_index=ignore_index
            ),
            s=pd.concat(
                [to_return.s, datatup_list[i].s], axis=axis, sort=False, ignore_index=ignore_index
            ),
            y=pd.concat(
                [to_return.y, datatup_list[i].y], axis=axis, sort=False, ignore_index=ignore_index
            ),
            name=to_return.name,
        )
    return to_return


def concat_tt(
    datatup_list: List[TestTuple], axis: str = "index", ignore_index: bool = False
) -> TestTuple:
    """Concatenate the test tuples in the given list"""

    to_return = TestTuple(x=datatup_list[0].x, s=datatup_list[0].s, name=datatup_list[0].name)
    for i in range(1, len(datatup_list)):
        to_return = TestTuple(
            x=pd.concat(
                [to_return.x, datatup_list[i].x], axis=axis, sort=False, ignore_index=ignore_index
            ),
            s=pd.concat(
                [to_return.s, datatup_list[i].s], axis=axis, sort=False, ignore_index=ignore_index
            ),
            name=to_return.name,
        )
    return to_return


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
    return DataTuple(
        x=train.x.iloc[:num], s=train.s.iloc[:num], y=train.y.iloc[:num], name=train.name
    )


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


class TrainTestPair(NamedTuple):
    train: DataTuple
    test: TestTuple
