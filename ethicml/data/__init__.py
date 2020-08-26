"""This module contains items related to data, such as raw csv's and data objects."""
from . import dataset, load, lookup, tabular_data, util, vision_data
from .dataset import *
from .load import *
from .lookup import *
from .tabular_data import *
from .util import *
from .vision_data import *

__all__ = ["available_tabular"]
for submodule in [dataset, load, lookup, tabular_data, util, vision_data]:
    __all__ += submodule.__all__  # type: ignore[attr-defined]


available_tabular = [
    adult.__name__,
    compas.__name__,
    credit.__name__,
    crime.__name__,
    german.__name__,
    health.__name__,
    sqf.__name__,
    toy.__name__,
]
