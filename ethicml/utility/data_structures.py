"""Data structures that are used throughout the code"""
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Tuple, List, Optional, NamedTuple, Callable, Iterator
from typing_extensions import Literal, Final

import pandas as pd
from pandas.testing import assert_index_equal

AxisType = Literal["columns", "index"]  # pylint: disable=invalid-name


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class TestTuple:
    """A tuple of dataframes for the features and the sensitive attribute"""

    x: pd.DataFrame  # features
    s: pd.DataFrame  # senstitive attributes
    name: Optional[str] = None  # name of the dataset

    def __iter__(self) -> Iterator[pd.DataFrame]:
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

    def replace(
        self,
        *,
        x: Optional[pd.DataFrame] = None,
        s: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
    ) -> "TestTuple":
        """Create a copy of the TestTuple but change the given values"""
        changes = {k: v for k, v in [("x", x), ("s", s), ("name", name)] if v is not None}
        return replace(self, **changes)


@dataclass(frozen=True)
class DataTupleValues:
    y: pd.DataFrame  # class labels


@dataclass(frozen=True)
class DataTuple(TestTuple, DataTupleValues):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels"""

    def __iter__(self) -> Iterator[pd.DataFrame]:
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

    def replace(
        self,
        *,
        x: Optional[pd.DataFrame] = None,
        s: Optional[pd.DataFrame] = None,
        y: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
    ) -> "DataTuple":
        """Create a copy of the DataTuple but change the given values"""
        changes = {k: v for k, v in [("x", x), ("s", s), ("y", y), ("name", name)] if v is not None}
        return replace(self, **changes)

    def apply_to_joined_df(self, mapper: Callable[[pd.DataFrame], pd.DataFrame]) -> "DataTuple":
        """Concatenate the dataframes in the DataTuple and then apply a function to it"""
        cols_x, cols_s, cols_y = self.x.columns, self.s.columns, self.y.columns
        joined = pd.concat([self.x, self.s, self.y], axis="columns", sort=False)
        joined = mapper(joined)
        result = self.replace(x=joined[cols_x], s=joined[cols_s], y=joined[cols_y])

        # assert that the columns haven't changed
        assert_index_equal(result.x.columns, cols_x)
        assert_index_equal(result.s.columns, cols_s)
        assert_index_equal(result.y.columns, cols_y)

        return result

    def get_subset(self, num: int = 500) -> "DataTuple":
        """Get the first elements of the dataset

        Args:
            num: how many samples to take for subset

        Returns:
            subset of training data
        """
        return self.replace(x=self.x.iloc[:num], s=self.s.iloc[:num], y=self.y.iloc[:num])


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
    return (train.write_as_feather(data_dir, "train"), test.write_as_feather(data_dir, "test"))


def concat_dt(
    datatup_list: List[DataTuple], axis: AxisType = "index", ignore_index: bool = False
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
    datatup_list: List[TestTuple], axis: AxisType = "index", ignore_index: bool = False
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


FairnessType = Literal["DP", "EqOp", "EqOd"]  # pylint: disable=invalid-name


def str_to_fair_type(fair_str: str) -> Optional[FairnessType]:
    """Convert a string to a fairness type or return None if not possible"""
    # this somewhat silly code is needed because mypy doesn't support narrowing to literals yet
    if fair_str == "DP":
        return "DP"
    if fair_str == "EqOd":
        return "EqOd"
    if fair_str == "EqOp":
        return "EqOp"
    return None


ClassifierType = Literal["LR", "SVM"]  # pylint: disable=invalid-name
ActivationType = Literal["identity", "logistic", "tanh", "relu"]  # pylint: disable=invalid-name


class TrainTestPair(NamedTuple):
    """2-Tuple of train and test data"""

    train: DataTuple
    test: TestTuple


class Results:
    """Container for results from `evaluate_models`"""

    columns: Final = ["dataset", "transform", "model", "repeat"]

    def __init__(self, data_frame: Optional[pd.DataFrame] = None):
        self._data: pd.DataFrame
        if data_frame is not None:
            # ensure correct index
            if data_frame.index.names != self.columns:
                self._data = data_frame.set_index(self.columns)
            else:
                self._data = data_frame
        else:
            self._data = pd.DataFrame(columns=self.columns).set_index(self.columns)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def __repr__(self) -> str:
        return repr(self._data)

    def __str__(self) -> str:
        return str(self._data)

    def append_df(self, data_frame: pd.DataFrame, prepend: bool = False) -> None:
        """Append (or prepend) a DataFrame to this object"""
        if data_frame.index.names != self.columns:
            data_frame = data_frame.set_index(self.columns)  # set correct index
        order = [data_frame, self._data] if prepend else [self._data, data_frame]
        # set sort=False so that the order of the columns is preserved
        self._data = pd.concat(order, sort=False, axis="index")

    def append_from_file(self, csv_file: Path, prepend: bool = False) -> bool:
        """Append results from a CSV file"""
        if csv_file.is_file():  # if file exists
            self.append_df(pd.read_csv(csv_file), prepend=prepend)
            return True
        return False

    def save_as_csv(self, file_path: Path) -> None:
        # `_data` has the multi index based on [dataset, transform, ...] so we have to reset that
        self._data.reset_index(drop=False).to_csv(file_path, index=False)

    @classmethod
    def from_file(cls, csv_file: Path) -> Optional["Results"]:
        """Load results from a CSV file that was created by `evaluate_models`

        Args:
            csv_file: path to a CSV file with results

        Returns:
            DataFrame if the file exists; None otherwise
        """
        if csv_file.is_file():
            return cls(pd.read_csv(csv_file))
        return None

    def map_over_index(
        self, mapper: Callable[[Tuple[str, str, str, str]], Tuple[str, str, str, str]]
    ) -> pd.DataFrame:
        """Change the values of the index with a transformation function"""
        results_mapped = self._data.copy()
        results_mapped.index = results_mapped.index.map(mapper)
        return results_mapped
