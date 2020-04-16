"""Transforms to be applied to a dataset."""

import torch
from torch import Tensor

__all__ = ["NoisyDequantize", "Quantize"]


class Transformation:
    """Base class for label-dependent augmentations.
    """

    def _transform(self, data: Tensor) -> Tensor:
        """Augment the input data
        Args:
            data: Tensor. Input data to be transformed.

        Returns:
            Tensor, transformed data
        """
        return data

    def __call__(self, data: Tensor) -> Tensor:
        """Calls the augment method on the the input data.

        Args:
            data: Tensor. Input data to be augmented.

        Returns:
            Tensor, augmented data
        """
        return self._transform(data)


class NoisyDequantize(Transformation):
    """Callable class for injecting noise into binned (e.g. image) data."""

    def __init__(self, n_bits_x: int = 8):
        """Createca NoisyQuantize object."""
        self.n_bins = 2 ** n_bits_x

    def _transform(self, data: Tensor) -> Tensor:
        return torch.clamp(data + (torch.rand_like(data) / self.n_bins), min=0, max=1)


class Quantize(Transformation):
    """Callable class that quantizes image data."""

    def __init__(self, n_bits_x: int = 8):
        """Create Quantize object."""
        self.n_bits_x = n_bits_x
        self.n_bins = 2 ** n_bits_x

    def _transform(self, data: Tensor) -> Tensor:
        if self.n_bits_x < 8:
            # for n_bits_x=5, this turns the range (0, 1) to (0, 32) and floors it
            # the exact value 32 will only appear if there was an exact 1 in `data`
            x = torch.floor(torch.clamp(data, 0, 1 - 1e-6) * self.n_bins)
            # re-normalize to between 0 and 1
            x = x / self.n_bins
        return x
