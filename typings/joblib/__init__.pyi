from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import ParamSpec, Self, TypeAlias

_CompressLevel: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def dump(
    value: Any,
    filename: Path,
    compress: Union[
        _CompressLevel, bool, Tuple[_CompressLevel, Literal["zlib", "gzip", "bz2", "lzma", "xz"]]
    ] = ...,
) -> List[str]: ...
def load(filename: Path) -> Any: ...

_T = TypeVar("_T")

class Parallel:
    def __init__(
        self,
        n_jobs: Optional[int] = ...,
        backend: Optional[Literal["loky", "multiprocessing", "threading"]] = ...,
        verbose: int = ...,
        timeout: Optional[float] = ...,
        *,
        prefer: Optional[Literal["processes", "threads"]] = ...,
        require: Optional[Literal["sharedmem"]] = ...,
    ): ...
    def __call__(self, iterable: Iterable[_DelayedResult[_T]]) -> List[_T]: ...
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
