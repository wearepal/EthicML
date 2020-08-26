"""In-process algorithms take training data and make predictions."""
from . import (
    agarwal_reductions,
    fairness_wo_demographics,
    in_algorithm,
    installed_model,
    kamiran,
    kamishima,
    logistic_regression,
    majority,
    manual,
    mlp,
    svm,
    svm_async,
    zafar,
)
from .agarwal_reductions import *
from .fairness_wo_demographics import *
from .in_algorithm import *
from .installed_model import *
from .kamiran import *
from .kamishima import *
from .logistic_regression import *
from .majority import *
from .manual import *
from .mlp import *
from .svm import *
from .svm_async import *
from .zafar import *

__all__ = []
for submodule in [
    agarwal_reductions,
    fairness_wo_demographics,
    in_algorithm,
    installed_model,
    kamiran,
    kamishima,
    logistic_regression,
    majority,
    manual,
    mlp,
    svm,
    svm_async,
    zafar,
]:
    __all__ += submodule.__all__  # type: ignore[attr-defined]
