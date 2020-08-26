"""This moodule contains algorithms that pre-process the data in some way."""

from . import adjust_labels, biased_split, domain_adaptation, feature_binning, scaling
from . import train_test_split as train_test_split_module
from .adjust_labels import *
from .biased_split import *
from .domain_adaptation import *
from .feature_binning import *
from .scaling import *
from .train_test_split import *

__all__ = []
for submodule in [
    adjust_labels,
    biased_split,
    domain_adaptation,
    feature_binning,
    scaling,
    train_test_split_module,
]:
    __all__ += submodule.__all__  # type: ignore[attr-defined]
