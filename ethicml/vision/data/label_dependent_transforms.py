"""Transformations that act differently depending on the label."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

__all__ = ["LdTransformation", "LdColorizer"]


class LdTransformation(ABC):
    """Base class for label-dependent augmentations."""

    @abstractmethod
    def _transform(self, data: Tensor, labels: Tensor) -> Tensor:
        """Augment the input data in a label-dependent fashion.

        Args:
            data: Input data to be augmented.
            labels: Labels on which the augmentations are conditioned.

        Returns:
            Augmented data.
        """
        ...

    def __call__(self, data: Tensor, labels: Tensor) -> Tensor:
        """Apply the augment method to the input data.

        Args:
            data: Input data to be augmented.
            labels: Labels on which the augmentations are conditioned.

        Returns:
            Augmented data.
        """
        return self._transform(data, labels)


class LdColorizer(LdTransformation):
    """Transform that colorizes images."""

    def __init__(
        self,
        scale: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        binarize: bool = False,
        background: bool = False,
        black: bool = True,
        seed: int = 42,
        greyscale: bool = False,
        color_indices: Optional[List[int]] = None,
    ):
        """Colorizes a grayscale image by sampling colors from multivariate normal distributions.

        The distribution is centered on predefined means and standard deviation determined by the
        scale argument.

        Args:
            min_val: Minimum value the input data can take (needed for clamping). Defaults to 0.
            max_val: Maximum value the input data can take (needed for clamping). Defaults to 1.
            scale: Standard deviation of the multivariate normal distributions from which
                   the colors are drawn. Lower values correspond to higher bias. Defaults to 0.02.
            binarize: Whether the binarize the grayscale data before colorisation. Defaults to False
            background: Whether to color the background instead of the foreground. Defaults to False
            black: Whether not to invert the black. Defaults to True.
            seed: Random seed used for sampling colors. Defaults to 42.
            greyscale: Whether to greyscale the colorised images. Defaults to False.
            color_indices: Choose specific colors if you don't need all 10
        """
        super(LdColorizer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.scale = scale * np.eye(3)
        self.binarize = binarize
        self.background = background
        self.black = black
        self.greyscale = greyscale

        # create a local random state that won't affect the global random state of the training
        self.random_state = np.random.RandomState(seed)

        colors = [
            (0, 255, 255),
            (0, 0, 255),  # blue
            (255, 0, 255),
            (0, 128, 0),
            (0, 255, 0),  # green
            (128, 0, 0),
            (0, 0, 128),
            (128, 0, 128),
            (255, 0, 0),  # red
            (255, 255, 0),
        ]  # yellow
        if color_indices:
            colors = [colors[i] for i in color_indices]

        self.palette = [np.divide(color, 255) for color in colors]

    def _sample_color(self, mean_color_values: np.ndarray) -> np.ndarray:
        return np.clip(self.random_state.multivariate_normal(mean_color_values, self.scale), 0, 1)

    def _transform(self, data: Tensor, labels: Tensor) -> Tensor:
        """Apply the transformation.

        Args:
            data: (Grayscale) data samples to be colorized.
            labels: Index (0-9) indicating the colour distribution from which to sample for each data
                    sample.

        Returns:
            Colorized tensor.
        """
        labels = labels.numpy()

        mean_values = []
        colors_per_sample: List[np.ndarray] = []
        for label in labels:
            mean_value = self.palette[label]
            mean_values.append(mean_value)
            colors_per_sample.append(self._sample_color(mean_value))

        if self.binarize:
            data = (data > 0.5).float()

        color_tensor = (
            torch.as_tensor(colors_per_sample, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        )  # type: ignore[call-arg]
        if self.background:
            if self.black:
                # colorful background, black digits
                transformed_data = (1 - data) * color_tensor  # type: ignore[operator]
            else:
                # colorful background, white digits
                transformed_data = torch.clamp(data + color_tensor, 0, 1)
        else:
            if self.black:
                # black background, colorful digits
                transformed_data = data * color_tensor
            else:
                # white background, colorful digits
                transformed_data = 1 - data * (1 - color_tensor)  # type: ignore[operator, assignment]

        if self.greyscale:
            transformed_data = transformed_data.mean(dim=1, keepdim=True)

        return transformed_data
