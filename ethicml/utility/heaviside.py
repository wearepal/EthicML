"""Implementation of Heaviside decision function."""
from __future__ import annotations

import numpy

from .activation import Activation

__all__ = ["Heaviside"]


class Heaviside(Activation):
    """Decision function that accepts predictions with score of 50% or above."""

    def apply(self, soft_output: numpy.ndarray) -> numpy.ndarray:
        """Apply the decision function to each element of an ndarray.

        :param soft_output: Soft predictions.
        :returns: Hard predictions.
        """

        def _heavi(x: float) -> int:
            return 1 if x >= 0.5 else 0

        return numpy.array([_heavi(x) for x in soft_output])

    def get_name(self) -> str:
        """Getter for name of decision function."""
        return "Heaviside"
