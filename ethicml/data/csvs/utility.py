"""Functions used in either generating data, or pre-processing raw data."""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """As there is no np.sigmoid."""
    return 1 / (1 + np.exp(-x))
