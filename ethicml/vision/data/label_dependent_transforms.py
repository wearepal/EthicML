"""Transformations that act differentlly depending on the label.
"""

import random
from typing import List

import numpy as np
import torch
from torch import Tensor
from skimage import color
from torchvision import transforms

__all__ = ["LdAugmentation", "LdColorizer"]


class LdAugmentation(torch.jit.ScriptModule):
    """Base class for label-dependent augmentations.
    """

    @torch.jit.script_method
    def _augment(self, data: Tensor, labels: Tensor) -> Tensor:
        """Augment the input data in a label-dependent fashion

        Args:
            data: Tensor. Input data to be augmented.
            labels: Tensor. Labels on which the augmentations are conditioned.

        Returns: Tensor. Augmented data.
        """
        return data

    def __call__(self, data: Tensor, labels: Tensor) -> Tensor:
        """Calls the augment method on the the input data.

        Args:
            data: Tensor. Input data to be augmented.
            labels: Tensor. Labels on which the augmentations are conditioned.

        Returns: Tensor. Augmented data.
        """
        return self._augment(data, labels)


class LdColorizer(LdAugmentation):

    __constants__ = ["color_space", "binarize", "black", "background", "seed"]

    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 1.0,
        scale: float = 0.02,
        binarize: bool = False,
        background: bool = False,
        black: bool = True,
        seed: bool = 42,
        greyscale: bool = False,
    ):
        """Colorizes a grayscale image by sampling colors from multivariate normal distributions
        centered on predefined means and standard deviation determined by the scale argument.

        Args:
            min_val (float, optional): Minimum value the input data can take (needed for clamping). Defaults to 0..
            max_val (float, optional): Maximum value the input data can take (needed for clamping). Defaults to 1..
            scale (float, optional): Standard deviation of the multivariate normal distributions from which
            the colors are drawn. Lower values correspond to higher bias. Defaults to 0.02.
            binarize (bool, optional): Whether the binarize the grayscale data before colorisation. Defaults to False.
            background (bool, optional): Whether to color the background instead of the foreground. Defaults to False.
            black (bool, optional): Whether not to invert the black. Defaults to True.
            seed (bool, optional): Random seed used for sampling colors. Defaults to 42.
            greyscale (bool, optional): Whether to greyscale the colorised images. Defaults to False.
        """
        super(LdColorizer, self).__init__()
        self.min_val
        self.max_val
        self.scale = scale
        self.binarize = binarize
        self.background = background
        self.black = black
        self.greyscale = greyscale

        # create a local random state that won't affect the global random state of the training
        self.random_state = np.random.RandomState(seed)

        self.color_space = color_space
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

        self.palette = [np.divide(color, 255) for color in colors]
        self.scale *= np.eye(3)

    def _sample_color(self, mean_color_values: np.ndarray) -> np.ndarray:
        if self.color_space == "hsv":
            return np.clip(self.random_state.normal(mean_color_values, self.scale), 0, 1)
        else:
            return np.clip(
                self.random_state.multivariate_normal(mean_color_values, self.scale), 0, 1
            )

    def _augment(self, data: Tensor, labels: Tensor) -> Tensor:
        """

        Args:
            data (Tensor): (Grayscale) data samples to be colorized.
            labels (Tensor): Index indicating the colour distribution
            from which to sample for each data sample.

        Returns:
            Tensor: Colorized tensor.
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

        color_tensor = Tensor(colors_per_sample).unsqueeze(-1).unsqueeze(-1)
        if self.background:
            if self.black:
                # colorful background, black digits
                augmented_data = (1 - data) * color_tensor
            else:
                # colorful background, white digits
                augmented_data = torch.clamp(data + color_tensor, 0, 1)
        else:
            if self.black:
                # black background, colorful digits
                augmented_data = data * color_tensor
            else:
                # white background, colorful digits
                augmented_data = 1 - data * (1 - color_tensor)

        if self.greyscale:
            augmented_data = augmented_data.mean(dim=1, keepdim=True)

        return augmented_data
