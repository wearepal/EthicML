from pathlib import Path as _Path
from typing import Union, Sequence, Tuple, List, Optional, overload
from typing_extensions import Literal

import numpy as _np

from .artist import Artist, Line2D, LineCollection, Rectangle
from .legend import Legend

_Data = Union[float, _np.ndarray[float], Sequence[float]]

_LegendLocation = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]

class Axes:
    def set_xlabel(self, xlabel: str) -> None: ...
    def set_ylabel(self, ylabel: str) -> None: ...
    def set_title(self, label: str, loc: Literal["left", "center", "right"] = ...) -> None: ...
    def grid(
        self,
        b: Optional[bool] = ...,
        which: Literal["major", "minor", "both"] = ...,
        axis: Literal["both", "x", "y"] = ...,
    ) -> None: ...
    def get_legend_handles_labels(
        self
    ) -> Tuple[List[Union[Artist, Tuple[Artist, ...]]], List[str]]: ...
    def legend(
        self,
        handles: Sequence[Union[Artist, Tuple[Artist, ...]]] = ...,
        labels: Sequence[str] = ...,
        loc: _LegendLocation = ...,
        bbox_to_anchor: Tuple[float, float] = ...,
    ) -> Legend: ...
    def errorbar(
        self,
        x: _Data,
        y: _Data,
        yerr: Optional[_Data] = ...,
        xerr: Optional[_Data] = ...,
        fmt: str = ...,
        ecolor: str = ...,
        elinewidth: float = ...,
        capsize: float = ...,
        barsabove: bool = ...,
        lolims: bool = ...,
        uplims: bool = ...,
        xlolims: bool = ...,
        xuplims: bool = ...,
        errorevery: int = ...,
        capthick: float = ...,
        color: str = ...,
        zorder: float = ...,
        label: str = ...,
        markersize: float = ...,
    ) -> Tuple[Line2D, Line2D, LineCollection]: ...
    def bar(
        self,
        x: _Data,
        height: _Data,
        width: _Data = ...,
        bottom: _Data = ...,
        *,
        align: Literal["center", "edge"] = ...,
        color: str = ...,
        edgecolor: str = ...,
        linewidth: float = ...,
        zorder: float = ...,
        hatch: str = ...,
        label: str = ...,
    ) -> Tuple[Rectangle, ...]: ...
    def plot(
        self,
        x: _Data,
        y: _Data,
        fmt: str = ...,
        scalex: bool = ...,
        scaley: bool = ...,
        color: str = ...,
        marker: str = ...,
        zorder: float = ...,
        markerfacecolor: str = ...,
        markersize: float = ...,
        label: str = ...,
    ): ...
    def set_xlim(self, xmin: float = ..., xmax: float = ..., auto: Optional[bool] = ...): ...
    def set_ylim(self, ymin: float = ..., ymax: float = ..., auto: Optional[bool] = ...): ...

class Figure:
    def savefig(
        self,
        fname: _Path,
        dpi: int = ...,
        bbox_extra_artists: Sequence[Artist] = ...,
        bbox_inches: Optional[Literal["tight"]] = ...,
    ) -> None: ...

@overload
def subplots(
    *,
    sharex: bool = ...,
    sharey: bool = ...,
    squeeze: Literal[True] = ...,
    dpi: int = ...,
    figsize: Tuple[float, float] = ...,
) -> Tuple[Figure, Axes]: ...
@overload
def subplots(
    nrows: int,
    sharex: bool = ...,
    sharey: bool = ...,
    squeeze: Literal[True] = ...,
    dpi: int = ...,
    figsize: Tuple[float, float] = ...,
) -> Tuple[Figure, List[Axes]]: ...
@overload
def subplots(
    nrows: int,
    ncols: int,
    sharex: bool = ...,
    sharey: bool = ...,
    squeeze: Literal[True] = ...,
    dpi: int = ...,
    figsize: Tuple[float, float] = ...,
) -> Tuple[Figure, List[List[Axes]]]: ...
@overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    squeeze: Literal[False],
    sharex: bool = ...,
    sharey: bool = ...,
    dpi: int = ...,
    figsize: Tuple[float, float] = ...,
) -> Tuple[Figure, List[List[Axes]]]: ...
def close(fig: Union[Figure, Literal["all"]]) -> None: ...
def clf() -> None: ...
