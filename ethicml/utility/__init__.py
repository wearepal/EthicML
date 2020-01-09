"""This module contains kind of useful things that don't really belong anywhere else (just yet)."""

from .activation import Activation
from .heaviside import Heaviside
from .data_structures import (
    ActivationType,
    DataTuple,
    FairnessType,
    PathTuple,
    Results,
    TestPathTuple,
    TestTuple,
    TrainTestPair,
    concat_dt,
    concat_tt,
    load_feather,
)
