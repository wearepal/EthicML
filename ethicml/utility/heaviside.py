"""
Implementation of Heaviside decision function
"""

import numpy
from .activation import Activation


class Heaviside(Activation):
    """Decision function that accepts predictions with score of 50% or above"""

    def apply(self, soft_output: numpy.array) -> numpy.array:
        def _heavi(x):
            return 1 if x >= 0.5 else 0
        return [_heavi(x) for x in soft_output]

    def get_name(self) -> str:
        return "Heaviside"
