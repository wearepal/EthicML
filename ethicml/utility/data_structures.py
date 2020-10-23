"""Data structures that are used throughout the code."""
import json
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal
from typing_extensions import Final, Literal

__all__ = [
    "ActivationType",
    "ClassifierType",
    "DataTuple",
    "FairnessType",
    "Prediction",
    "Results",
    "ResultsAggregator",
    "SoftPrediction",
    "TestTuple",
    "TrainTestPair",
    "aggregate_results",
    "concat_dt",
    "concat_tt",
    "filter_and_map_results",
    "filter_results",
    "make_results",
    "map_over_results_index",
]

AxisType = Literal["columns", "index"]  # pylint: disable=invalid-name


class TestTuple:
    """A tuple of dataframes for the features and the sensitive attribute."""

    def __init__(self, x: pd.DataFrame, s: pd.DataFrame, name: Optional[str] = None):
        """Make a TestTuple."""
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

    def to_npz(self, data_path: Path) -> None:
        """Save TestTuple as an npz file."""
        write_as_npz(
            data_path,
            dict(x=self.x, s=self.s),
            dict(name=np.array(self.name if self.name is not None else "")),
        )

    @classmethod
    def from_npz(cls, data_path: Path) -> "TestTuple":
        """Load test tuple from npz file."""
        with data_path.open("rb") as data_file:
            data = np.load(data_file)
            name = data["name"].item()
            return cls(
                x=pd.DataFrame(data["x"], columns=data["x_names"]),
                s=pd.DataFrame(data["s"], columns=data["s_names"]),
                name=name if name else None,
            )


class DataTuple(TestTuple):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels.

    Args:
        x: input features
        s: sensitive attributes
        y: class labels
        name: optional name of the dataset
    """

    def __init__(
        self, x: pd.DataFrame, s: pd.DataFrame, y: pd.DataFrame, name: Optional[str] = None
    ):
        """Make a DataTuple."""
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

    def to_npz(self, data_path: Path) -> None:
        """Save DataTuple as an npz file."""
        write_as_npz(
            data_path,
            dict(x=self.x, s=self.s, y=self.y),
            dict(name=np.array(self.name if self.name is not None else "")),
        )

    @classmethod
    def from_npz(cls, data_path: Path) -> "DataTuple":
        """Load data tuple from npz file."""
        with data_path.open("rb") as data_file:
            data = np.load(data_file)
            name = data["name"].item()
            return cls(
                x=pd.DataFrame(data["x"], columns=data["x_names"]),
                s=pd.DataFrame(data["s"], columns=data["s_names"]),
                y=pd.DataFrame(data["y"], columns=data["y_names"]),
                name=name if name else None,
            )


class Prediction:
    """Prediction of an algorithm."""

    def __init__(self, hard: pd.Series, info: Optional[Dict[str, float]] = None):
        """Make a prediction obj."""
        assert isinstance(hard, pd.Series), "please use pd.Series"
        self._hard = hard
        self._info = info if info is not None else {}

    def __len__(self) -> int:
        """Length of the predictions object."""
        return len(self._hard)

    @property
    def hard(self) -> pd.Series:
        """Hard predictions (e.g. 0 and 1)."""
        return self._hard

    @property
    def info(self) -> Dict[str, float]:
        """Additional info about the prediction."""
        return self._info

    @staticmethod
    def from_npz(npz_path: Path) -> "Prediction":
        """Load prediction from npz file."""
        info = None
        if (npz_path.parent / "info.json").exists():
            with open(npz_path.parent / "info.json") as json_file:
                info = json.load(json_file)
        with npz_path.open("rb") as npz_file:
            data = np.load(npz_file)
            if "soft" in data:
                return SoftPrediction(soft=pd.Series(np.squeeze(data["soft"])), info=info)
            return Prediction(hard=pd.Series(np.squeeze(data["hard"])), info=info)

    def to_npz(self, npz_path: Path) -> None:
        """Save prediction as npz file."""
        if self.info:
            for v in self.info.values():
                assert isinstance(v, float), "Info must be Dict[str, float]"
            json_path = npz_path.parent / 'info.json'
            with open(json_path, 'w') as json_file:
                json.dump(self.info, json_file)

        np.savez(npz_path, hard=self.hard.to_numpy())


class SoftPrediction(Prediction):
    """Prediction of an algorithm that makes soft predictions."""

    def __init__(self, soft: pd.Series, info: Optional[Dict[str, float]] = None):
        """Make a soft prediction object."""
        super().__init__(hard=soft.ge(0.5).astype(int), info=info)
        self._soft = soft

    @property
    def soft(self) -> pd.Series:
        """Soft predictions (e.g. 0.2 and 0.8)."""
        return self._soft


def write_as_npz(
    data_path: Path, data: Dict[str, pd.DataFrame], extra: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """Write the given dataframes to an npz file."""
    extra = extra or {}
    as_numpy = {entry: values.to_numpy() for entry, values in data.items()}
    column_names = {
        f"{entry}_names": np.array(values.columns.tolist()) for entry, values in data.items()
    }
    np.savez(data_path, **as_numpy, **column_names, **extra)


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


class TrainValPair(NamedTuple):
    """2-Tuple of train and test data."""

    train: DataTuple
    validation: DataTuple


Results = NewType("Results", pd.DataFrame)  # Container for results from `evaluate_models`

RESULTS_COLUMNS: Final = ["dataset", "transform", "model", "split_id"]


def make_results(data_frame: Union[None, pd.DataFrame, Path] = None) -> Results:
    """Initialise Results object.

    You should always use this function instead of using the "constructor" directly, because this
    function checks whether the columns are correct.
    """
    if isinstance(data_frame, Path):
        data_frame = pd.read_csv(data_frame)
    if data_frame is not None:
        # ensure correct index
        if data_frame.index.names != RESULTS_COLUMNS:  # type: ignore[comparison-overlap]
            return Results(data_frame.set_index(RESULTS_COLUMNS))
        else:
            return Results(data_frame)
    else:
        return Results(pd.DataFrame(columns=RESULTS_COLUMNS).set_index(RESULTS_COLUMNS))


class ResultsAggregator:
    """Aggregate results."""

    def __init__(self, initial: Optional[pd.DataFrame] = None):
        """Init results aggregator obj."""
        self._results = make_results(initial)

    @property
    def results(self) -> Results:
        """Results object over which this class is aggregating."""
        return self._results

    def append_df(self, data_frame: pd.DataFrame, prepend: bool = False) -> None:
        """Append (or prepend) a DataFrame to this object."""
        if data_frame.index.names != RESULTS_COLUMNS:  # type: ignore[comparison-overlap]
            data_frame = data_frame.set_index(RESULTS_COLUMNS)  # set correct index
        order = [data_frame, self.results] if prepend else [self.results, data_frame]
        # set sort=False so that the order of the columns is preserved
        self._results = Results(pd.concat(order, sort=False, axis="index"))

    def append_from_csv(self, csv_file: Path, prepend: bool = False) -> bool:
        """Append results from a CSV file."""
        if csv_file.is_file():  # if file exists
            self.append_df(pd.read_csv(csv_file), prepend=prepend)
            return True
        return False

    def save_as_csv(self, file_path: Path) -> None:
        """Save to csv."""
        # `results` has the multi index based on [dataset, transform, ...] so we have to reset that
        self.results.reset_index(drop=False).to_csv(file_path, index=False)


def map_over_results_index(
    results: Results, mapper: Callable[[Tuple[str, str, str, str]], Tuple[str, str, str, str]]
) -> Results:
    """Change the values of the index with a transformation function."""
    results_mapped = results.copy()
    results_mapped.index = results_mapped.index.map(mapper)
    return make_results(results_mapped)


def filter_results(
    results: Results, values: Iterable, index: Literal["dataset", "transform", "model"] = "model"
) -> Results:
    """Filter the entries based on the given values."""
    return Results(results.loc[results.index.get_level_values(index).isin(list(values))])


def filter_and_map_results(results: Results, mapping: Mapping[str, str]) -> Results:
    """Filter entries and change the index with a mapping."""
    return map_over_results_index(
        filter_results(results, mapping),
        lambda index: (index[0], index[1], mapping[index[2]], index[3]),
    )


def aggregate_results(
    results: Results, metrics: List[str], aggregator: Union[str, Tuple[str, ...]] = ("mean", "std")
) -> pd.DataFrame:
    """Aggregate results over the repeats."""
    return results.groupby(["dataset", "transform", "model"]).agg(aggregator)[metrics]
