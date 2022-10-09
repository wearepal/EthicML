"""Data structures that are used throughout the code."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import json
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Final,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    TypeVar,
    Union,
    final,
)
from typing_extensions import TypeAlias

import numpy as np
from numpy import typing as npt
import pandas as pd
from ranzen import StrEnum

__all__ = [
    "ClassifierType",
    "DataTuple",
    "EvalTuple",
    "FairnessType",
    "HyperParamType",
    "HyperParamValue",
    "KernelType",
    "LabelTuple",
    "ModelType",
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
    def replace_data(self: _S, data: pd.DataFrame, name: str | None = None) -> _S:
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
        return self.replace_data(data=self.data.loc[self.s == s])

    @final
    def __len__(self) -> int:
        """Number of entries in the underlying data."""
        return len(self.data)


@dataclass
class SubgroupTuple(SubsetMixin):
    """A tuple of dataframes for the features and the sensitive attribute."""

    __slots__ = ("data", "s_column", "s_in_x", "name")
    data: pd.DataFrame
    s_column: str
    s_in_x: bool
    name: Optional[str]

    def __post_init__(self) -> None:
        assert self.s_column in self.data.columns, f"column {self.s_column} not present"

    @classmethod
    def from_df(
        cls, *, x: pd.DataFrame, s: pd.Series[int], name: str | None = None
    ) -> SubgroupTuple:
        """Make a SubgroupTuple."""
        s_column = s.name
        assert isinstance(s_column, str)
        assert len(x) == len(s), "data has to have the same length"
        if s_column in x.columns:
            # sometimes s can appear in x, but we should ensure they're actually the same
            pd.testing.assert_series_equal(s, x[s_column])
            data = x
            s_in_x = True
        else:
            data = pd.concat([x, s], axis="columns", sort=False)
            s_in_x = False
        return cls(data=data, s_column=s_column, s_in_x=s_in_x, name=name)

    @property
    def x(self) -> pd.DataFrame:
        """Getter for property x."""
        if self.s_in_x:
            return self.data
        return self.data.drop(self.s_column, inplace=False, axis="columns")

    def __iter__(self) -> Iterator[pd.DataFrame | pd.Series]:
        """Iterator of ``self.x`` and ``self.s``."""
        return iter([self.x, self.s])

    def replace(
        self, *, x: pd.DataFrame | None = None, s: pd.Series | None = None
    ) -> SubgroupTuple:
        """Create a copy of the SubgroupTuple but change the given values."""
        return SubgroupTuple.from_df(
            x=x if x is not None else self.x, s=s if s is not None else self.s, name=self.name
        )

    def replace_data(self, data: pd.DataFrame, name: str | None = None) -> SubgroupTuple:
        """Make a copy of the DataTuple but change the underlying data."""
        assert self.s_column in data.columns, f"column {self.s_column} not present"
        return SubgroupTuple(
            data=data,
            s_column=self.s_column,
            s_in_x=self.s_in_x,
            name=self.name if name is None else name,
        )

    def rename(self, name: str) -> SubgroupTuple:
        """Change only the name."""
        return SubgroupTuple(data=self.data, s_column=self.s_column, s_in_x=self.s_in_x, name=name)

    def save_to_file(self, data_path: Path) -> None:
        """Save SubgroupTuple as an npz file.

        :param data_path: Path to save the npz file.
        """
        write_as_npz(
            data_path,
            dict(x=self.x, s=self.s),
            dict(name=np.array(self.name if self.name is not None else "")),
        )

    @classmethod
    def from_file(cls, data_path: Path) -> SubgroupTuple:
        """Load test tuple from npz file.

        :param data_path: Path to load the npz file.
        :returns: A :class:`SubgroupTuple` with the loaded data.
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

    __slots__ = ("data", "s_column", "y_column", "s_in_x", "name")
    data: pd.DataFrame
    s_column: str
    y_column: str
    s_in_x: bool
    name: Optional[str]

    def __post_init__(self) -> None:
        assert self.s_column in self.data.columns, f"column {self.s_column} not present"
        assert self.y_column in self.data.columns, f"column {self.y_column} not present"

    @classmethod
    def from_df(
        cls, *, x: pd.DataFrame, s: pd.Series[int], y: pd.Series[int], name: str | None = None
    ) -> DataTuple:
        """Make a DataTuple."""
        s_column = s.name
        y_column = y.name
        assert isinstance(s_column, str) and isinstance(y_column, str)
        assert y_column not in x.columns, f"overlapping columns in `x` and `y`: {y_column}"
        assert len(x) == len(s) == len(y), "data has to have the same length"
        if s_column in x.columns:
            # sometimes s can appear in x, but we should ensure they're actually the same
            pd.testing.assert_series_equal(s, x[s_column])
            data = pd.concat([x, y], axis="columns", sort=False)
            s_in_x = True
        else:
            data = pd.concat([x, s, y], axis="columns", sort=False)
            s_in_x = False
        return cls(data=data, s_column=s_column, y_column=y_column, s_in_x=s_in_x, name=name)

    @property
    def x(self) -> pd.DataFrame:
        """Getter for property x."""
        if self.s_in_x:
            return self.data.drop(self.y_column, inplace=False, axis="columns")
        return self.data.drop([self.s_column, self.y_column], inplace=False, axis="columns")

    @property
    def y(self) -> pd.Series[int]:
        """Getter for property y."""
        return self.data[self.y_column]

    def __iter__(self) -> Iterator[pd.DataFrame | pd.Series]:
        """Iterator of ``self.x``, ``self.s`` and ``self.y``."""
        return iter([self.x, self.s, self.y])

    def remove_y(self) -> SubgroupTuple:
        """Convert the DataTuple instance to a SubgroupTuple instance."""
        return SubgroupTuple(
            data=self.data.drop(self.y_column, inplace=False, axis="columns"),
            s_column=self.s_column,
            s_in_x=self.s_in_x,
            name=self.name,
        )

    def replace(
        self,
        *,
        x: pd.DataFrame | None = None,
        s: pd.Series | None = None,
        y: pd.Series | None = None,
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
        return DataTuple(
            data=self.data,
            s_column=self.s_column,
            y_column=self.y_column,
            s_in_x=self.s_in_x,
            name=name,
        )

    def replace_data(self, data: pd.DataFrame, name: str | None = None) -> DataTuple:
        """Make a copy of the DataTuple but change the underlying data."""
        assert self.s_column in data.columns, f"column {self.s_column} not present"
        assert self.y_column in data.columns, f"column {self.y_column} not present"
        return DataTuple(
            data=data,
            s_column=self.s_column,
            y_column=self.y_column,
            s_in_x=self.s_in_x,
            name=self.name if name is None else name,
        )

    def apply_to_joined_df(self, mapper: Callable[[pd.DataFrame], pd.DataFrame]) -> DataTuple:
        """Concatenate the dataframes in the DataTuple and then apply a function to it.

        :param mapper: A function that takes a dataframe and returns a dataframe.
        :returns: The transformed :class:`DataTuple`.
        """
        return self.replace_data(data=mapper(self.data))

    def save_to_file(self, data_path: Path) -> None:
        """Save DataTuple as an npz file.

        :param data_path: Path to the npz file.
        """
        write_as_npz(
            data_path,
            dict(x=self.x, s=self.s, y=self.y),
            dict(name=np.array(self.name if self.name is not None else "")),
        )

    @classmethod
    def from_file(cls, data_path: Path) -> DataTuple:
        """Load data tuple from npz file.

        :param data_path: Path to the npz file.
        :returns: A :class:`DataTuple` with the loaded data.
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
    name: Optional[str]

    def __post_init__(self) -> None:
        assert self.s_column in self.data.columns, f"column {self.s_column} not present"
        assert self.y_column in self.data.columns, f"column {self.y_column} not present"

    @classmethod
    def from_df(
        cls, *, s: pd.Series[int], y: pd.Series[int], name: str | None = None
    ) -> LabelTuple:
        """Make a LabelTuple."""
        s_column = s.name
        y_column = y.name
        assert isinstance(s_column, str) and isinstance(y_column, str)
        assert s_column != y_column, f"name of `s` and `y` is the same: {s_column}"
        assert len(s) == len(y), "data has to have the same length"
        it: Iterable[pd.Series] = [s, y]
        return cls(
            data=pd.concat(it, axis="columns", sort=False),
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

    def __iter__(self) -> Iterator[pd.DataFrame | pd.Series]:
        """Iterator of ``self.x`` and ``self.y``."""
        return iter([self.s, self.y])

    def replace(self, *, s: pd.Series | None = None, y: pd.Series | None = None) -> LabelTuple:
        """Create a copy of the LabelTuple but change the given values."""
        return LabelTuple.from_df(
            s=s if s is not None else self.s, y=y if y is not None else self.y, name=self.name
        )

    def rename(self, name: str) -> LabelTuple:
        """Change only the name."""
        return LabelTuple(data=self.data, s_column=self.s_column, y_column=self.y_column, name=name)

    def replace_data(self, data: pd.DataFrame, name: str | None = None) -> LabelTuple:
        """Make a copy of the LabelTuple but change the underlying data."""
        assert self.s_column in data.columns, f"column {self.s_column} not present"
        assert self.y_column in data.columns, f"column {self.y_column} not present"
        return LabelTuple(
            data=data,
            s_column=self.s_column,
            y_column=self.y_column,
            name=self.name if name is None else name,
        )


TestTuple: TypeAlias = Union[SubgroupTuple, DataTuple]
"""Union of :class:`SubgroupTuple` and :class:`DataTuple`."""

EvalTuple: TypeAlias = Union[LabelTuple, DataTuple]
"""Union of :class:`LabelTuple` and :class:`DataTuple`."""

T = TypeVar("T", SubgroupTuple, DataTuple)


class Prediction:
    """Prediction of an algorithm."""

    def __init__(self, hard: pd.Series, info: HyperParamType | None = None):
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
        :returns: The requested subset as a new ``Prediction`` object.
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
    def from_file(npz_path: Path) -> Prediction:
        """Load prediction from npz file.

        :param npz_path: Path to the npz file.
        :returns: A :class:`Prediction` object with the loaded data.
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

    def save_to_file(self, npz_path: Path) -> None:
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

    def __init__(self, soft: np.ndarray, info: HyperParamType | None = None):
        """Make a soft prediction object."""
        super().__init__(hard=pd.Series(soft.argmax(axis=1).astype(int), name="hard"), info=info)
        self._soft = soft

    @property
    def soft(self) -> np.ndarray:
        """Soft predictions (e.g. 0.2 and 0.8)."""
        return self._soft


def write_as_npz(
    data_path: Path,
    data: dict[str, pd.DataFrame | pd.Series],
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    """Write the given dataframes to an npz file.

    :param data_path: Path to the npz file.
    :param data: Dataframes to save.
    :param extra: Extra data to save. (Default: None)
    """
    extra = extra or {}
    as_numpy = {entry: values.to_numpy() for entry, values in data.items()}
    column_names: dict[str, np.ndarray] = {
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
    :returns: The concatenated data tuple.
    """
    data: pd.DataFrame = pd.concat(
        [dt.data for dt in datatup_list], axis="index", sort=False, ignore_index=ignore_index
    )
    return datatup_list[0].replace_data(data)


class FairnessType(StrEnum):
    """Fairness type."""

    dp = auto()
    """Demographic parity."""
    eq_opp = auto()
    """Equality of Opportunity."""
    eq_odds = auto()
    """Equalized Odds."""


class ClassifierType(StrEnum):
    """Classifier type."""

    lr = auto()
    """Logistic Regression."""
    svm = auto()
    """Support Vector Machine."""
    gbt = auto()
    """Gradient Boosting."""


class TrainTestPair(NamedTuple):
    """2-Tuple of train and test data."""

    train: DataTuple
    test: SubgroupTuple


class TrainValPair(NamedTuple):
    """2-Tuple of train and test data."""

    train: DataTuple
    test: DataTuple


Results = NewType("Results", pd.DataFrame)
"""Container for results from :func:`~ethicml.run.evaluate_models`."""


RESULTS_COLUMNS: Final = ["dataset", "scaler", "transform", "model", "split_id"]


def make_results(data_frame: None | pd.DataFrame | Path = None) -> Results:
    """Initialise Results object.

    You should always use this function instead of using the "constructor" directly, because this
    function checks whether the columns are correct.

    :param data_frame: A dataframe to use for initialization. (Default: None)
    :returns: An initialised :class:`Results` object.
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

    def __init__(self, initial: pd.DataFrame | None = None):
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
            # set the correct index
            data_frame = data_frame.set_index(RESULTS_COLUMNS)
        order = [data_frame, self.results] if prepend else [self.results, data_frame]
        # set sort=False so that the order of the columns is preserved
        self._results = Results(pd.concat(order, sort=False, axis="index"))

    def append_from_csv(self, csv_file: Path, prepend: bool = False) -> bool:
        """Append results from a CSV file.

        :param csv_file: Path to the CSV file.
        :param prepend:  (Default: False)
        :returns: ``True`` if the file existed and was succesfully loaded; ``False`` otherwise.
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
    mapper: Callable[[tuple[str, str, str, str, str]], tuple[str, str, str, str, str]],
) -> Results:
    """Change the values of the index with a transformation function."""
    results_mapped = results.copy()
    results_mapped.index = results_mapped.index.map(mapper)
    return make_results(results_mapped)


def filter_results(
    results: Results,
    values: Iterable,
    index: str | PandasIndex = "model",
) -> Results:
    """Filter the entries based on the given values.

    :param results: Results object to filter.
    :param values: Values to filter on.
    :param index: Index to filter on. (Default: "model")
    :returns: The filtered results.
    """
    if isinstance(index, str):
        index = PandasIndex(index)
    return Results(results.loc[results.index.get_level_values(index.value).isin(list(values))])


def filter_and_map_results(results: Results, mapping: Mapping[str, str]) -> Results:
    """Filter entries and change the index with a mapping.

    :param results: Results object to filter.
    :param mapping: Mapping from old index to new index.
    :returns: The filtered and mapped results.
    """
    return map_over_results_index(
        filter_results(results, mapping),
        lambda index: (index[0], index[1], index[2], mapping[index[3]], index[4]),
    )


def aggregate_results(
    results: Results, metrics: list[str], aggregator: str | tuple[str, ...] = ("mean", "std")
) -> pd.DataFrame:
    """Aggregate results over the repeats.

    :param results: Results object containing the results to aggregate.
    :param metrics: Metrics used for aggregation.
    :param aggregator: Aggregator to use. The aggreators are the ones used in pandas.
        (Default: ("mean", "std"))
    :returns: The aggregated results as a ``pd.DataFrame``.
    """
    return results.groupby(["dataset", "scaler", "transform", "model"]).agg(aggregator)[metrics]


class KernelType(StrEnum):
    """Values for SVM Kernel."""

    linear = auto()
    """Linear kernel."""
    poly = auto()
    """Polynomial kernel."""
    rbf = auto()
    """Radial basis function kernel."""
    sigmoid = auto()
    """Sigmoid kernel."""


class ModelType(StrEnum):
    """What to use as the underlying model for the fairness method."""

    deep = auto()
    """Deep neural network."""
    linear = auto()
    """Linear model."""


HyperParamValue: TypeAlias = Union[bool, int, float, str]
HyperParamType: TypeAlias = Dict[str, HyperParamValue]
