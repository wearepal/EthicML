"""Data objects for Tabular data."""
# these imports need to be renamed because they clash with the functions of the same name
from . import adult as adult_module
from . import compas as compas_module
from . import credit as credit_module
from . import crime as crime_module
from . import german as german_module
from . import health as health_module
from . import non_binary_toy
from . import sqf as sqf_module
from . import synthetic as synthetic_module
from . import toy as toy_module
from .adult import *
from .compas import *
from .credit import *
from .crime import *
from .german import *
from .health import *
from .non_binary_toy import *
from .sqf import *
from .synthetic import *
from .toy import *

__all__ = []
for submodule in [
    adult_module,
    compas_module,
    credit_module,
    crime_module,
    german_module,
    health_module,
    non_binary_toy,
    sqf_module,
    synthetic_module,
    toy_module,
]:
    __all__ += submodule.__all__  # type: ignore[attr-defined]
