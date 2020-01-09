"""Data structures that are used throughout the code."""
from pathlib import Path
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from typing_extensions import Literal, Final

import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal

__all__ = [
    "ActivationType",
    "ClassifierType",
    "DataTuple",
    "FairnessType",
    "PathTuple",
    "Prediction",
    "Results",
    "SoftPrediction",
    "TestPathTuple",
    "TestTuple",
    "TrainTestPair",
    "concat_dt",
    "concat_tt",
    "load_prediction",
    "save_helper",
]

AxisType = Literal["columns", "index"]  # pylint: disable=invalid-name


class TestTuple:
    """A tuple of dataframes for the features and the sensitive attribute."""

    def __init__(self, x: pd.DataFrame, s: pd.DataFrame, name: Optional[str] = None):
        """Init function for TestTuple object."""
        self.__x: pd.DataFrame = x
        self.__s: pd.DataFrame = s
        self.__name: Optional[str] = name

    @property
    def x(self) -> pd.DataFrame:
        """Getter for property x."""
        return self.__x

    @property
    def s(self) -> pd.DataFrame:
        """Getter for property s."""
        return self.__s

    @property
    def name(self) -> Optional[str]:
        """Getter for name property."""
        return self.__name

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Overwrite magic method __iter__."""
        return iter([self.x, self.s])

    def write_as_feather(self, data_dir: Path, identifier: str) -> "TestPathTuple":
        """Write the TestTuple to Feather files and return the file paths as a TestPathTuple.

        Args:
            data_dir: directory where the files should be stored
            identifier: a string that ideally identifies the file uniquely, can be a UUID
        Returns:
            tuple of paths
        """
        # create the directory if it doesn't already exist
        data_dir.mkdir(parents=True, exist_ok=True)

        # SUGGESTION: maybe the file names should be completely random to avoid collisions
        data_path = data_dir / f"data_{identifier}.npz"
        return TestPathTuple(
            data_path=save_helper(data_path, dict(x=self.x, s=self.s)),
            name=self.name if self.name is not None else "",
        )

    def replace(
        self,
        *,
        x: Optional[pd.DataFrame] = None,
        s: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
    ) -> "TestTuple":
        """Create a copy of the TestTuple but change the given values."""
        return TestTuple(
            x=x if x is not None else self.x,
            s=s if s is not None else self.s,
            name=name if name is not None else self.name,
        )


class DataTuple(TestTuple):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels."""

    def __init__(
        self, x: pd.DataFrame, s: pd.DataFrame, y: pd.DataFrame, name: Optional[str] = None
    ):
        """Init for DataTuple Class."""
        super().__init__(x=x, s=s, name=name)
        self.__y: pd.DataFrame = y

    @property
    def y(self) -> pd.DataFrame:
        """Getter for property y."""
        return self.__y

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Overwrite __iter__ magic method."""
        return iter([self.x, self.s, self.y])

    def __len__(self) -> int:
        """Overwrite __len__ magic method."""
        len_x = len(self.x)
        assert len_x == len(self.s) and len_x == len(self.y)
        return len_x

    def remove_y(self) -> TestTuple:
        """Convert the DataTuple instance to a TestTuple instance."""
        return TestTuple(x=self.x, s=self.s, name=self.name)

    def write_as_feather(self, data_dir: Path, identifier: str) -> "PathTuple":
        """Write the DataTuple to Feather files and return the file paths as a PathTuple.

        Args:
            data_dir: directory where the files should be stored
            identifier: a string that ideally identifies the file uniquely, can be a UUID
        Returns:
            tuple of paths
        """
        # create the directory if it doesn't already exist
        data_dir.mkdir(parents=True, exist_ok=True)

        # SUGGESTION: maybe the file names should be completely random to avoid collisions
        data_path = data_dir / f"data_{identifier}.npz"
        return PathTuple(
            data_path=save_helper(data_path, dict(x=self.x, s=self.s, y=self.y)),
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
        """Create a copy of the DataTuple but change the given values."""
        return DataTuple(
            x=x if x is not None else self.x,
            s=s if s is not None else self.s,
            y=y if y is not None else self.y,
            name=name if name is not None else self.name,
        )

    def apply_to_joined_df(self, mapper: Callable[[pd.DataFrame], pd.DataFrame]) -> "DataTuple":
        """Concatenate the dataframes in the DataTuple and then apply a function to it."""
        self.x.columns = self.x.columns.astype(str)
        cols_x, cols_s, cols_y = self.x.columns, self.s.columns, self.y.columns
        joined = pd.concat([self.x, self.s, self.y], axis="columns", sort=False)
        assert len(joined) == len(self), "something went wrong while concatenating"
        joined = mapper(joined)
        result = self.replace(x=joined[cols_x], s=joined[cols_s], y=joined[cols_y])

        # assert that the columns haven't changed
        assert_index_equal(result.x.columns, cols_x)
        assert_index_equal(result.s.columns, cols_s)
        assert_index_equal(result.y.columns, cols_y)

        return result

    def get_subset(self, num: int = 500) -> "DataTuple":
        """Get the first elements of the dataset.

        Args:
            num: how many samples to take for subset

        Returns:
            subset of training data
        """
        return self.replace(x=self.x.iloc[:num], s=self.s.iloc[:num], y=self.y.iloc[:num])


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class TestPathTuple:
    """For algorithms that run in their own process, we pass around paths to the data."""

    data_path: Path  # path to file with features
    name: str  # name of the dataset

    def load_from_feather(self) -> TestTuple:
        """Load a dataframe from a feather file."""
        data = np.load(self.data_path)
        return TestTuple(
            x=pd.DataFrame(data['x'], columns=data['x_names']),
            s=pd.DataFrame(data['s'], columns=data['s_names']),
            name=self.name if self.name else None,
        )


@dataclass(frozen=True)  # "frozen" means the objects are immutable
class PathTuple(TestPathTuple):
    """For algorithms that run in their own process, we pass around paths to the data."""

    def load_from_feather(self) -> DataTuple:
        """Load a dataframe from a feather file."""
        data = np.load(self.data_path)
        return DataTuple(
            x=pd.DataFrame(data['x'], columns=data['x_names']),
            s=pd.DataFrame(data['s'], columns=data['s_names']),
            y=pd.DataFrame(data['y'], columns=data['y_names']),
            name=self.name if self.name else None,
        )


class Prediction:
    """Prediction of an algorithm."""

    def __init__(self, hard: pd.Series, info: Optional[Dict[str, float]] = None):
        """Init the prediction class."""
        self._hard = hard
        self._info = info if info is not None else {}

    @property
    def hard(self) -> pd.Series:
        """Hard predictions (e.g. 0 and 1)."""
        return self._hard

    @property
    def info(self) -> Dict[str, float]:
        """Additional info about the prediction."""
        return self._info


class SoftPrediction(Prediction):
    """Prediction of an algorithm that makes soft predictions."""

    def __init__(self, soft: pd.Series, info: Optional[Dict[str, float]] = None):
        """Init the soft prediction class."""
        super().__init__(hard=soft.ge(0.5).astype(int), info=info)
        self._soft = soft

    @property
    def soft(self) -> pd.Series:
        """Soft predictions (e.g. 0.2 and 0.8)."""
        return self._soft


def save_helper(data_path: Path, data: Dict[str, pd.DataFrame]) -> Path:
    # write the file
    as_numpy = {entry: values.to_numpy() for entry, values in data.items()}
    column_names = {
        f"{entry}_names": np.array(values.columns.tolist()) for entry, values in data.items()
    }
    np.savez(data_path, **as_numpy, **column_names)
    return data_path


def write_as_feather(
    train: DataTuple, test: TestTuple, data_dir: Path
) -> (Tuple[PathTuple, TestPathTuple]):
    """Write the given DataTuple to Feather files and return the file paths as PathTuples.

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


def concat_dt(
    datatup_list: Sequence[DataTuple], axis: AxisType = "index", ignore_index: bool = False
) -> DataTuple:
    """Concatenate the data tuples in the given list."""
    return DataTuple(
        x=pd.concat(
            [dt.x for dt in datatup_list], axis=axis, sort=False, ignore_index=ignore_index
        ),
        s=pd.concat(
            [dt.s for dt in datatup_list], axis=axis, sort=False, ignore_index=ignore_index
        ),
        y=pd.concat(
            [dt.y for dt in datatup_list], axis=axis, sort=False, ignore_index=ignore_index
        ),
        name=datatup_list[0].name,
    )


def concat_tt(
    datatup_list: List[TestTuple], axis: AxisType = "index", ignore_index: bool = False
) -> TestTuple:
    """Concatenate the test tuples in the given list."""
    return TestTuple(
        x=pd.concat(
            [dt.x for dt in datatup_list], axis=axis, sort=False, ignore_index=ignore_index
        ),
        s=pd.concat(
            [dt.s for dt in datatup_list], axis=axis, sort=False, ignore_index=ignore_index
        ),
        name=datatup_list[0].name,
    )


def load_prediction(output_path: Path) -> Prediction:
    """Load a prediction from a path."""
    with output_path.open("rb") as file_obj:
        df = np.load(file_obj)
    preds = df["pred"]
    return Prediction(hard=pd.Series(preds))


FairnessType = Literal["DP", "EqOp", "EqOd"]  # pylint: disable=invalid-name


def str_to_fair_type(fair_str: str) -> Optional[FairnessType]:
    """Convert a string to a fairness type or return None if not possible."""
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
    """2-Tuple of train and test data."""

    train: DataTuple
    test: TestTuple


class Results:
    """Container for results from `evaluate_models`."""

    columns: Final = ["dataset", "transform", "model", "split_id"]

    def __init__(self, data_frame: Optional[pd.DataFrame] = None):
        """Initialise Results object."""
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
        """Getter for data property."""
        return self._data

    def __repr__(self) -> str:
        """Overwrite __repr__ magic method."""
        return repr(self._data)

    def __str__(self) -> str:
        """Overwrite __str__ magic method."""
        return str(self._data)

    def __len__(self) -> int:
        """Overwrite __len__ magic method."""
        return len(self._data)

    def append_df(self, data_frame: pd.DataFrame, prepend: bool = False) -> None:
        """Append (or prepend) a DataFrame to this object."""
        if data_frame.index.names != self.columns:
            data_frame = data_frame.set_index(self.columns)  # set correct index
        order = [data_frame, self._data] if prepend else [self._data, data_frame]
        # set sort=False so that the order of the columns is preserved
        self._data = pd.concat(order, sort=False, axis="index")

    def append_from_file(self, csv_file: Path, prepend: bool = False) -> bool:
        """Append results from a CSV file."""
        if csv_file.is_file():  # if file exists
            self.append_df(pd.read_csv(csv_file), prepend=prepend)
            return True
        return False

    def save_as_csv(self, file_path: Path) -> None:
        """Save to csv."""
        # `_data` has the multi index based on [dataset, transform, ...] so we have to reset that
        self._data.reset_index(drop=False).to_csv(file_path, index=False)

    @classmethod
    def from_file(cls, csv_file: Path) -> Optional["Results"]:
        """Load results from a CSV file that was created by `evaluate_models`.

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
    ) -> "Results":
        """Change the values of the index with a transformation function."""
        results_mapped = self._data.copy()
        results_mapped.index = results_mapped.index.map(mapper)
        return Results(results_mapped)

    def filter(
        self, values: Iterable, index: Literal["dataset", "transform", "model"] = "model"
    ) -> "Results":
        """Filter the entries based on the given values."""
        return Results(self._data.loc[self._data.index.get_level_values(index).isin(list(values))])

    def query(self, query_str: str) -> "Results":
        """Query the underlying dataframe."""
        return Results(self._data.query(query_str))

    def filter_and_map(self, mapping: Mapping[str, str]) -> "Results":
        """Filter entries and change the index with a mapping."""
        return self.filter(mapping).map_over_index(
            lambda index: (index[0], index[1], mapping[index[2]], index[3])
        )

    def aggregate(
        self, metrics: List[str], aggregator: Union[str, Tuple[str]] = ("mean", "std")
    ) -> pd.DataFrame:
        """Aggregate results over the repeats."""
        return self._data.groupby(["dataset", "transform", "model"]).agg(aggregator)[metrics]
