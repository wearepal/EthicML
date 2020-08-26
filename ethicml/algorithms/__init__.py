"""Definitions of algorithms."""
from . import algorithm_base, inprocess, preprocess
from .algorithm_base import *
from .inprocess import *
from .preprocess import *

__all__ = []
for submodule in [algorithm_base, inprocess, preprocess]:
    __all__ += submodule.__all__  # type: ignore[attr-defined]
