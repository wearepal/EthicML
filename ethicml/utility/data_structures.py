"""Data structures that are used throughout the code."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
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
    TypeVar,
    Union,
)
from typing_extensions import Final, Literal, TypeAlias, final

import numpy as np
import pandas as pd
from numpy import typing as npt
from ranzen import enum_name_str

__all__ = [
    "ClassifierType",
    "DataTuple",
    "EvalTuple",
    "FairnessType",
    "HyperParamType",
    "HyperParamValue",
    "KernelType",
    "Prediction",
    "Results",
    "ResultsAggregator",
    "SoftPrediction",
    "SubgroupTuple",
    "TestTuple",
    "TrainTestPair",
    "TrainValPair",
    "aggregate_results",
    "concat",
    "filter_and_map_results",
    "filter_results",
    "make_results",
    "map_over_results_index",
]

AxisType: TypeAlias = Literal["columns", "index"]  # pylint: disable=invalid-name


class PandasIndex(Enum):
    """Enum for indexing the results."""

    DATASET = "dataset"
    SCALER = "scaler"
    TRANSFORM = "transform"
    MODEL = "model"


_S = TypeVar("_S", bound="SubsetMixin")


class SubsetMixin(ABC):
    """Mixin that provides methods for getting subsets."""

    # the mixin assumes the presence of these attributes
    data: pd.DataFrame
    s_column: str

    @abstractmethod
    def replace_data(self: _S, data: pd.DataFrame) -> _S:
        """Make a copy of the container but change the underlying data."""

    @property
    @final
    def s(self) -> pd.Series[int]:
        """Getter for property s."""
        return self.data[self.s_column]

    @final
    def get_n_samples(self: _S, num: int = 500) -> _S:
        """Get the first elements of the dataset.

        :param num: How many samples to take for subset. (Default: 500)
        :returns: Subset of training data.
        """
        return self.replace_data(data=self.data.iloc[:num])

    @final
    def get_s_subset(self: _S, s: int) -> _S:
        """Return a subset of the DataTuple where S=s."""
        return self.replace_data(data=self.data[self.s == s])

    @final
    def __len__(self) -> int:
        """Overwrite __len__ magic method."""
        return len(self.data)


@dataclass
class SubgroupTuple(SubsetMixin):
    """A tuple of dataframes for the features and the sensitive attribute."""

    __slots__ = ("data", "s_column", "name")
    data: pd.DataFrame
    s_column: str
    name: str | None

    def __post_init__(self) -> None:
        assert self.s_column in self.data.columns, f"column {self.s_column} not present"

    @classmethod
    def from_df(
        cls, *, x: pd.DataFrame, s: pd.Series[int], name: Optional[str] = None
    ) -> SubgroupTuple:
        """Make a SubgroupTuple."""
        s_column = s.name
        assert isinstance(s_column, str)
        assert len(x) == len(s), "data has to have the same length"
        return cls(data=pd.concat([x, s], axis="columns", sort=False), s_column=s_column, name=name)

    @property
    def x(self) -> pd.DataFrame:
        """Getter for property x."""
        return self.data.drop(self.s_column, inplace=False, axis="columns")

    def __iter__(self) -> Iterator[Union[pd.DataFrame, pd.Series]]:
        """Overwrite magic method __iter__."""
        return iter([self.x, self.s])

    def replace(
        self, *, x: Optional[pd.DataFrame] = None, s: Optional[pd.Series] = None
    ) -> SubgroupTuple:
        """Create a copy of the SubgroupTuple but change the given values."""
        return SubgroupTuple.from_df(
            x=x if x is not None else self.x, s=s if s is not None else self.s, name=self.name
        )

    def replace_data(self, data: pd.DataFrame) -> SubgroupTuple:
        """Make a copy of the DataTuple but change the underlying data."""
        assert self.s_column in data.columns, f"column {self.s_column} not present"
        return SubgroupTuple(data=data, s_column=self.s_column, name=self.name)

    def rename(self, name: str) -> SubgroupTuple:
        """Change only the name."""
        return SubgroupTuple(data=self.data, s_column=self.s_column, name=name)

    def to_npz(self, data_path: Path) -> None:
        """Save SubgroupTuple as an npz file.

        :param data_path: Path to save the npz file.
        """
        write_as_npz(
            data_path,
            dict(x=self.x, s=self.s),
            dict(name=np.array(self.name if self.name is not None else "")),
        )

    @classmethod
    def from_npz(cls, data_path: Path) -> SubgroupTuple:
        """Load test tuple from npz file.

        :param data_path: Path to load the npz file.
        """
        with data_path.open("rb") as data_file:
            data = np.load(data_file)
            name = data["name"].item()
            return cls.from_df(
                x=pd.DataFrame(data["x"], columns=data["x_names"]),
                s=pd.Series(data["s"], name=data["s_names"][0]),
                name=name or None,
            )


@dataclass
class DataTuple(SubsetMixin):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels."""

    __slots__ = ("data", "s_column", "y_column", "name")
    data: pd.DataFrame
    s_column: str
    y_column: str
    name: str | None

    def __post_init__(self) -> None:
        assert self.s_column in self.data.columns, f"column {self.s_column} not present"
        assert self.y_column in self.data.columns, f"column {self.y_column} not present"

    @classmethod
    def from_df(
        cls, *, x: pd.DataFrame, s: pd.Series[int], y: pd.Series[int], name: Optional[str] = None
    ) -> DataTuple:
        """Make a DataTuple."""
        s_column = s.name
        y_column = y.name
        assert isinstance(s_column, str) and isinstance(y_column, str)
        assert len(x) == len(s) == len(y), "data has to have the same length"
        return cls(
            data=pd.concat([x, s, y], axis="columns", sort=False),
            s_column=s_column,
            y_column=y_column,
            name=name,
        )

    @property
    def x(self) -> pd.DataFrame:
        """Getter for property x."""
        return self.data.drop([self.s_column, self.y_column], inplace=False, axis="columns")

    @property
    def y(self) -> pd.Series[int]:
        """Getter for property y."""
        return self.data[self.y_column]

    def __iter__(self) -> Iterator[Union[pd.DataFrame, pd.Series]]:
        """Overwrite magic method __iter__."""
        return iter([self.x, self.s, self.y])

    def remove_y(self) -> SubgroupTuple:
        """Convert the DataTuple instance to a SubgroupTuple instance."""
        return SubgroupTuple(
            data=self.data.drop(self.y_column, inplace=False, axis="columns"),
            s_column=self.s_column,
            name=self.name,
        )

    def replace(
        self,
        *,
        x: Optional[pd.DataFrame] = None,
        s: Optional[pd.Series] = None,
        y: Optional[pd.Series] = None,
    ) -> DataTuple:
        """Create a copy of the DataTuple but change the given values."""
        return DataTuple.from_df(
            x=x if x is not None else self.x,
            s=s if s is not None else self.s,
            y=y if y is not None else self.y,
            name=self.name,
        )

    def rename(self, name: str) -> DataTuple:
        """Change only the name."""
        return DataTuple(data=self.data, s_column=self.s_column, y_column=self.y_column, name=name)

    def replace_data(self, data: pd.DataFrame) -> DataTuple:
        """Make a copy of the DataTuple but change the underlying data."""
        assert self.s_column in data.columns, f"column {self.s_column} not present"
        assert self.y_column in data.columns, f"column {self.y_column} not present"
        return DataTuple(data=data, s_column=self.s_column, y_column=self.y_column, name=self.name)

    def apply_to_joined_df(self, mapper: Callable[[pd.DataFrame], pd.DataFrame]) -> DataTuple:
        """Concatenate the dataframes in the DataTuple and then apply a function to it.

        :param mapper: A function that takes a dataframe and returns a dataframe.
        """
        return self.replace_data(data=mapper(self.data))

    def to_npz(self, data_path: Path) -> None:
        """Save DataTuple as an npz file.

        :param data_path: Path to the npz file.
        """
        write_as_npz(
            data_path,
            dict(x=self.x, s=self.s, y=self.y),
            dict(name=np.array(self.name if self.name is not None else "")),
        )

    @classmethod
    def from_npz(cls, data_path: Path) -> DataTuple:
        """Load data tuple from npz file.

        :param data_path: Path to the npz file.
        """
        with data_path.open("rb") as data_file:
            data = np.load(data_file)
            name = data["name"].item()
            return cls.from_df(
                x=pd.DataFrame(data["x"], columns=data["x_names"]),
                s=pd.Series(data["s"], name=data["s_names"][0]),
                y=pd.Series(data["y"], name=data["y_names"][0]),
                name=name or None,
            )


@dataclass
class LabelTuple(SubsetMixin):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels."""

    __slots__ = ("data", "s_column", "y_column", "name")
    data: pd.DataFrame
    s_column: str
    y_column: str
    name: str | None

    def __post_init__(self) -> None:
        assert self.s_column in self.data.columns, f"column {self.s_column} not present"
        assert self.y_column in self.data.columns, f"column {self.y_column} not present"

    @classmethod
    def from_df(
        cls, *, s: pd.Series[int], y: pd.Series[int], name: Optional[str] = None
    ) -> LabelTuple:
        """Make a LabelTuple."""
        s_column = s.name
        y_column = y.name
        assert isinstance(s_column, str) and isinstance(y_column, str)
        assert len(s) == len(y), "data has to have the same length"
        return cls(
            data=pd.concat([s, y], axis="columns", sort=False),  # type: ignore[arg-type]
            s_column=s_column,
            y_column=y_column,
            name=name,
        )

    @classmethod
    def from_np(
        cls, *, s: npt.NDArray, y: npt.NDArray, s_name: str = "s", y_name: str = "y"
    ) -> LabelTuple:
        """Create a LabelTuple from numpy arrays."""
        s_pd = pd.Series(s, name=s_name)
        y_pd = pd.Series(y, name=y_name)
        assert len(s_pd) == len(y_pd), "data has to have the same length"
        return cls(
            data=pd.concat([s_pd, y_pd], axis="columns", sort=False),
            s_column=s_name,
            y_column=y_name,
            name=None,
        )

    @property
    def y(self) -> pd.Series[int]:
        """Getter for property y."""
        return self.data[self.y_column]

    def __iter__(self) -> Iterator[Union[pd.DataFrame, pd.Series]]:
        """Overwrite magic method __iter__."""
        return iter([self.s, self.y])

    def replace(
        self, *, s: Optional[pd.Series] = None, y: Optional[pd.Series] = None
    ) -> LabelTuple:
        """Create a copy of the LabelTuple but change the given values."""
        return LabelTuple.from_df(
            s=s if s is not None else self.s, y=y if y is not None else self.y, name=self.name
        )

    def rename(self, name: str) -> LabelTuple:
        """Change only the name."""
        return LabelTuple(data=self.data, s_column=self.s_column, y_column=self.y_column, name=name)

    def replace_data(self, data: pd.DataFrame) -> LabelTuple:
        """Make a copy of the LabelTuple but change the underlying data."""
        assert self.s_column in data.columns, f"column {self.s_column} not present"
        assert self.y_column in data.columns, f"column {self.y_column} not present"
        return LabelTuple(data=data, s_column=self.s_column, y_column=self.y_column, name=self.name)


TestTuple: TypeAlias = Union[SubgroupTuple, DataTuple]
EvalTuple: TypeAlias = Union[LabelTuple, DataTuple]
T = TypeVar("T", SubgroupTuple, DataTuple)


class Prediction:
    """Prediction of an algorithm."""

    def __init__(self, hard: pd.Series, info: Optional[HyperParamType] = None):
        """Make a prediction obj."""
        assert isinstance(hard, pd.Series), "please use pd.Series"
        self._hard = hard
        self._info = info if info is not None else {}

    @classmethod
    def from_np(cls, preds: npt.NDArray) -> Prediction:
        """Construct a prediction object from a numpy array."""
        return cls(hard=pd.Series(preds))

    def __len__(self) -> int:
        """Length of the predictions object."""
        return len(self._hard)

    def get_s_subset(self, s_data: pd.Series, s: int) -> Prediction:
        """Return a subset of the DataTuple where S=s.

        :param s_data: Dataframe with the s-values.
        :param s: S-value to get the subset for.
        """
        return Prediction(hard=self.hard[s_data == s])

    @property
    def hard(self) -> pd.Series:
        """Hard predictions (e.g. 0 and 1)."""
        return self._hard

    @property
    def info(self) -> HyperParamType:
        """Additional info about the prediction."""
        return self._info

    @staticmethod
    def from_npz(npz_path: Path) -> Prediction:
        """Load prediction from npz file.

        :param npz_path: Path to the npz file.
        """
        info = None
        if (npz_path.parent / "info.json").exists():
            with open(npz_path.parent / "info.json", encoding="utf-8") as json_file:
                info = json.load(json_file)
        with npz_path.open("rb") as npz_file:
            data = np.load(npz_file)
            if "soft" in data:
                return SoftPrediction(soft=np.squeeze(data["soft"]), info=info)
            return Prediction(hard=pd.Series(np.squeeze(data["hard"])), info=info)

    def to_npz(self, npz_path: Path) -> None:
        """Save prediction as npz file.

        :param npz_path: Path to the npz file.
        """
        if self.info:
            for v in self.info.values():
                assert isinstance(v, float), "Info must be Dict[str, float]"
            json_path = npz_path.parent / "info.json"
            with open(json_path, "w") as json_file:
                json.dump(self.info, json_file)

        np.savez(npz_path, hard=self.hard.to_numpy())


class SoftPrediction(Prediction):
    """Prediction of an algorithm that makes soft predictions."""

    def __init__(self, soft: np.ndarray, info: Optional[HyperParamType] = None):
        """Make a soft prediction object."""
        super().__init__(hard=pd.Series(soft.argmax(axis=1).astype(int), name="hard"), info=info)
        self._soft = soft

    @property
    def soft(self) -> np.ndarray:
        """Soft predictions (e.g. 0.2 and 0.8)."""
        return self._soft


def write_as_npz(
    data_path: Path,
    data: Dict[str, Union[pd.DataFrame, pd.Series]],
    extra: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write the given dataframes to an npz file.

    :param data_path: Path to the npz file.
    :param data: Dataframes to save.
    :param extra: Extra data to save. (Default: None)
    """
    extra = extra or {}
    as_numpy = {entry: values.to_numpy() for entry, values in data.items()}
    column_names: Dict[str, np.ndarray] = {
        f"{entry}_names": np.array(values.columns.tolist())
        if isinstance(values, pd.DataFrame)
        else np.array([values.name])
        for entry, values in data.items()
    }

    np.savez(data_path, **as_numpy, **column_names, **extra)


def concat(
    datatup_list: Sequence[T],
    ignore_index: bool = False,
) -> T:
    """Concatenate the data tuples in the given list.

    :param datatup_list: List of data tuples to concatenate.
    :param ignore_index: Ignore the index of the dataframes. (Default: False)
    """
    data: pd.DataFrame = pd.concat(
        [dt.data for dt in datatup_list], axis="index", sort=False, ignore_index=ignore_index
    )
    return datatup_list[0].replace_data(data)


@enum_name_str
class FairnessType(Enum):
    """Fairness type."""

    dp = auto()
    eq_opp = auto()
    eq_odds = auto()


@enum_name_str
class ClassifierType(Enum):
    """Classifier type."""

    lr = auto()
    svm = auto()
    gbt = auto()


class TrainTestPair(NamedTuple):
    """2-Tuple of train and test data."""

    train: DataTuple
    test: SubgroupTuple


class TrainValPair(NamedTuple):
    """2-Tuple of train and test data."""

    train: DataTuple
    test: DataTuple


Results = NewType("Results", pd.DataFrame)  # Container for results from `evaluate_models`

RESULTS_COLUMNS: Final = ["dataset", "scaler", "transform", "model", "split_id"]


def make_results(data_frame: Union[None, pd.DataFrame, Path] = None) -> Results:
    """Initialise Results object.

    You should always use this function instead of using the "constructor" directly, because this
    function checks whether the columns are correct.

    :param data_frame: A dataframe to use for initialization. (Default: None)
    """
    if isinstance(data_frame, Path):
        data_frame = pd.read_csv(data_frame)
    if data_frame is None:
        return Results(pd.DataFrame(columns=RESULTS_COLUMNS).set_index(RESULTS_COLUMNS))
    # ensure correct index
    if data_frame.index.names != RESULTS_COLUMNS:
        return Results(data_frame.set_index(RESULTS_COLUMNS))
    else:
        return Results(data_frame)


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
        """Append (or prepend) a DataFrame to this object.

        :param data_frame: DataFrame to append.
        :param prepend: Whether to prepend or append the dataframe. (Default: False)
        """
        if data_frame.index.names != RESULTS_COLUMNS:
            data_frame = data_frame.set_index(RESULTS_COLUMNS)  # set correct index
        order = [data_frame, self.results] if prepend else [self.results, data_frame]
        # set sort=False so that the order of the columns is preserved
        self._results = Results(pd.concat(order, sort=False, axis="index"))

    def append_from_csv(self, csv_file: Path, prepend: bool = False) -> bool:
        """Append results from a CSV file.

        :param csv_file: Path to the CSV file.
        :param prepend:  (Default: False)
        """
        if csv_file.is_file():  # if file exists
            self.append_df(pd.read_csv(csv_file), prepend=prepend)
            return True
        return False

    def save_as_csv(self, file_path: Path) -> None:
        """Save to csv.

        :param file_path: Path to the CSV file.
        """
        # `results` has the multi index based on [dataset, transform, ...] so we have to reset that
        self.results.reset_index(drop=False).to_csv(file_path, index=False)


def map_over_results_index(
    results: Results,
    mapper: Callable[[Tuple[str, str, str, str, str]], Tuple[str, str, str, str, str]],
) -> Results:
    """Change the values of the index with a transformation function."""
    results_mapped = results.copy()
    results_mapped.index = results_mapped.index.map(mapper)
    return make_results(results_mapped)


def filter_results(
    results: Results,
    values: Iterable,
    index: Union[str, PandasIndex] = "model",
) -> Results:
    """Filter the entries based on the given values.

    :param results: Results object to filter.
    :param values: Values to filter on.
    :param index: Index to filter on. (Default: "model")
    """
    if isinstance(index, str):
        index = PandasIndex(index)
    return Results(results.loc[results.index.get_level_values(index.value).isin(list(values))])


def filter_and_map_results(results: Results, mapping: Mapping[str, str]) -> Results:
    """Filter entries and change the index with a mapping.

    :param results: Results object to filter.
    :param mapping: Mapping from old index to new index.
    """
    return map_over_results_index(
        filter_results(results, mapping),
        lambda index: (index[0], index[1], index[2], mapping[index[3]], index[4]),
    )


def aggregate_results(
    results: Results, metrics: List[str], aggregator: Union[str, Tuple[str, ...]] = ("mean", "std")
) -> pd.DataFrame:
    """Aggregate results over the repeats.

    :param results: Results object containing the results to aggregate.
    :param metrics: Metrics used for aggregation.
    :param aggregator: Aggregator to use. The aggreators are the ones used in pandas.
        (Default: ("mean", "std"))
    """
    return results.groupby(["dataset", "scaler", "transform", "model"]).agg(aggregator)[metrics]  # type: ignore[arg-type]


@enum_name_str
class KernelType(Enum):
    """Values for SVM Kernel."""

    linear = auto()
    poly = auto()
    rbf = auto()
    sigmoid = auto()


HyperParamValue: TypeAlias = Union[bool, int, float, str, FairnessType, KernelType]
HyperParamType: TypeAlias = Dict[str, HyperParamValue]
