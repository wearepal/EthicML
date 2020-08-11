"""Class for loading images.

Modifies the Pytorch vision dataset by enabling the use of sensitive attributes
and biased subset sampling.
"""
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset

from ethicml.utility import DataTuple

__all__ = ["TorchImageDataset"]


class TorchImageDataset(VisionDataset):
    """Image dataset for pytorch."""

    def __init__(
        self,
        data: DataTuple,
        root: Path,
        map_to_binary: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """Large-scale CelebFaces Attributes (CelebA) Dataset.

        <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
        Adapted from torchvision.datasets to enable the loading of data triplets and biased/unbiased
        subsets while removing superfluous (for our purposes) elements of the dataset (e.g. facial
        landmarks).

        Args:
            data: Data tuple with x containing the filepaths to the generated faces images.
            root: Root directory where images are downloaded to.
            map_to_binary: if True, convert labels of {-1, 1} to {0, 1}
            transform: A function/transform that  takes in an PIL image and returns a transformed
                       version. E.g, `transforms.ToTensor`
            target_transform: A function/transform that takes in the target and transforms it.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        s = data.s
        self.s_dim = 1
        y = data.y

        if map_to_binary:
            s = (s + 1) // 2  # map from {-1, 1} to {0, 1}
            y = (y + 1) // 2  # map from {-1, 1} to {0, 1}

        self.x: np.ndarray[np.str_] = data.x["filename"].to_numpy()
        self.s = torch.as_tensor(s.to_numpy())
        self.y = torch.as_tensor(y.to_numpy())

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Fetch the data sample at the given index.

        Args:
            index (int): Index of the sample to be loaded in.

        Returns:
            Tuple[1]: Tuple containing the sample along
            with its sensitive and target attribute labels.
        """
        x = Image.open(str(self.root / self.x[index]))
        s = self.s[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, s, y

    def __len__(self) -> int:
        """Length (sample count) of the dataset.

        Returns:
            Integer indicating the length of the dataset.
        """
        return self.s.size(0)
