from collections.abc import Callable, Iterable
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, Literal, TypeVar, overload
from typing_extensions import ParamSpec, Self, TypeAliasType

_CompressLevel = TypeAliasType("_CompressLevel", Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

def dump(
    value: Any,
    filename: Path,
    compress: _CompressLevel
    | bool
    | tuple[_CompressLevel, Literal["zlib", "gzip", "bz2", "lzma", "xz"]] = ...,
) -> list[str]: ...
def load(filename: Path) -> Any: ...

_T = TypeVar("_T")

class Parallel:
    def __init__(
        self,
        n_jobs: int | None = ...,
        backend: Literal["loky", "multiprocessing", "threading"] | None = ...,
        verbose: int = ...,
        timeout: float | None = ...,
        *,
        prefer: Literal["processes", "threads"] | None = ...,
        require: Literal["sharedmem"] | None = ...,
    ): ...
    def __call__(self, iterable: Iterable[_DelayedResult[_T]]) -> list[_T]: ...
    def print_progress(self) -> None: ...
    def __enter__(self) -> Self: ...
    @overload
    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None: ...
    @overload
    def __exit__(
        self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None: ...
    def retrieve(self) -> None: ...

class _DelayedResult(Generic[_T]): ...

_P = ParamSpec("_P")

def delayed(function: Callable[_P, _T]) -> Callable[_P, _DelayedResult[_T]]: ...
