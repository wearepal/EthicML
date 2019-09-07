from typing import (
    Any,
    Tuple,
    List,
    Union,
    Callable,
    Dict,
    Iterator,
    Type,
    IO,
    Optional,
    overload,
    Sequence,
    Generic,
    TypeVar,
)
from typing_extensions import Literal
from pathlib import Path as _Path
import numpy as _np

_T = TypeVar('_T', str, int)
_AxisType = Literal["columns", "index"]

class Index(Generic[_T]):
    # magic methods
    def __eq__(self, other: object) -> Series: ...  # type: ignore
    def __getitem__(self, idx: int) -> _T: ...
    def __iter__(self) -> Iterator: ...
    #
    # properties
    @property
    def values(self) -> _np.ndarray: ...
    #
    # methods
    def astype(self, dtype: Type) -> Index: ...
    def get_level_values(self, level: str) -> Index: ...
    def to_frame(self) -> DataFrame: ...

class Series:
    # magic methods
    def __and__(self, other: Series) -> Series: ...
    def __eq__(self, other: object) -> Series: ...  # type: ignore
    @overload
    def __getitem__(self, idx: Union[List[str], Index[int], Series, slice]) -> Series: ...
    @overload
    def __getitem__(self, idx: int) -> float: ...
    def __truediv__(self, other: object) -> Series: ...
    #
    # properties
    @property
    def index(self) -> Index[int]: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def values(self) -> _np.ndarray: ...
    #
    # methods
    def all(self, axis: int = ..., bool_only: bool = ...) -> bool: ...
    def count(self) -> int: ...
    def max(self) -> float: ...
    def mean(self) -> float: ...
    def min(self) -> float: ...
    def nunique(self) -> int: ...
    def replace(self, to_replace: int, value: int, inplace: bool) -> None: ...
    def std(self) -> float: ...
    def to_numpy(self) -> _np.ndarray: ...
    def unique(self) -> List[float]: ...
    def value_counts(self) -> Series: ...

_ListLike = Union[_np.ndarray, Series, List, Dict[str, _np.ndarray]]

class DataFrame:
    def __init__(
        self,
        data: Optional[Union[_ListLike, DataFrame]] = ...,
        columns: Optional[Union[List[str], Index]] = ...,
        index: Optional[Union[_np.ndarray, Index]] = ...,
    ): ...
    #
    # magic methods
    @overload
    def __getitem__(self, idx: str) -> Series: ...
    @overload
    def __getitem__(self, idx: Union[List[str], Index]) -> DataFrame: ...
    def __len__(self) -> int: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    #
    # properties
    @property
    def columns(self) -> Index[str]: ...
    @columns.setter  # setter needs to be right next to getter; otherwise mypy complains
    def columns(self, cols: Union[List[str], Index[str]]) -> None: ...
    @property
    def iloc(self) -> _iLocIndexer: ...
    @property
    def index(self) -> Index[int]: ...
    @property
    def loc(self) -> _LocIndexer: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def size(self) -> int: ...
    @property
    def values(self) -> _np.ndarray: ...
    #
    # methods
    def append(
        self, s: Dict[str, Any], ignore_index: bool = ..., sort: bool = ...
    ) -> DataFrame: ...
    def apply(self, f: Callable) -> DataFrame: ...
    def copy(self) -> DataFrame: ...
    def count(self) -> Series: ...
    def drop(self, index: Union[List[str], Index], axis: _AxisType = ...) -> DataFrame: ...
    def head(self, n: int) -> DataFrame: ...
    def nunique(self) -> Series: ...
    def query(self, expr: str) -> DataFrame: ...
    def rename(self, mapper: Callable, axis: _AxisType = ...) -> DataFrame: ...
    def replace(self, a: float, b: float) -> DataFrame: ...
    def reset_index(self, drop: bool) -> DataFrame: ...
    def sample(self, frac: float, random_state: int, replace: bool = ...) -> DataFrame: ...
    def set_index(self, index: List[str]) -> DataFrame: ...
    @overload
    def sort_values(
        self, by: List[str], inplace: Literal[True], axis: _AxisType = ..., ascending: bool = ...
    ) -> None: ...
    @overload
    def sort_values(
        self,
        by: List[str],
        inplace: Optional[Literal[False]] = ...,
        axis: _AxisType = ...,
        ascending: bool = ...,
    ) -> DataFrame: ...
    def to_csv(self, filename: _Path, index: bool = ...) -> None: ...
    def to_feather(self, filename: _Path) -> None: ...
    def to_numpy(self) -> _np.ndarray: ...

class _iLocIndexer:
    @overload
    def __getitem__(self, idx: int) -> Series: ...
    @overload
    def __getitem__(
        self, idx: Union[slice, Sequence[int], _np.ndarray[int], Index[int]]
    ) -> DataFrame: ...
    @overload
    def __setitem__(self, idx: int, value: Series) -> None: ...
    @overload
    def __setitem__(
        self, idx: Union[slice, Sequence[int], _np.ndarray[int], Index[int]], value: DataFrame
    ) -> None: ...

class _LocIndexer:
    def __getitem__(self, idx: Union[Series, _np.ndarray[bool]]) -> DataFrame: ...
    def __setitem__(self, idx: Union[Series, _np.ndarray[bool]], value: Series) -> None: ...

def concat(
    dataframes: List[DataFrame],
    axis: _AxisType = ...,
    sort: Optional[bool] = ...,
    ignore_index: bool = ...,
) -> DataFrame: ...
def cut(arr: _np.ndarray, bins: int) -> Tuple[Union[Series, _np.ndarray], _np.ndarray]: ...
def get_dummies(df: Union[DataFrame, Series]) -> DataFrame: ...
def isnull(df: Union[DataFrame, Series]) -> _np.ndarray: ...
def read_csv(p: _Path) -> DataFrame: ...
def read_feather(p: Union[_Path, IO]) -> DataFrame: ...
