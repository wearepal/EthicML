"""Pre-process algorithms take the training data and transform it."""
from . import beutel, calders, pre_algorithm, upsampler, vfae, zemel
from .beutel import *
from .calders import *
from .pre_algorithm import *
from .upsampler import *
from .vfae import *
from .zemel import *

__all__ = []
for submodule in [beutel, calders, pre_algorithm, upsampler, vfae, zemel]:
    __all__ += submodule.__all__  # type: ignore[attr-defined]
