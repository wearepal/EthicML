"""This module contains items related to data, such as raw csv's and data objects."""
from .dataset import *
from .load import *
from .lookup import *
from .tabular_data.adult import *
from .tabular_data.compas import *
from .tabular_data.credit import *
from .tabular_data.crime import *
from .tabular_data.german import *
from .tabular_data.health import *
from .tabular_data.non_binary_toy import *
from .tabular_data.sqf import *
from .tabular_data.synthetic import *
from .tabular_data.toy import *
from .util import *
from .vision_data.celeba import *
from .vision_data.genfaces import *

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
