"""Vision datasets."""
from . import celeba as celeba_module
from . import genfaces as genfaces_module
from .celeba import *
from .genfaces import *

__all__ = []
for submodule in [celeba_module, genfaces_module]:
    __all__ += submodule.__all__  # type: ignore[attr-defined]
