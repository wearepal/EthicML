"""Class for loading CelebA.

Modifies the Pytorch CelebA dataset by enabling the use of sensitive attributes
and biased subset sampling.
"""
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive

from ethicml.common import implements
from ethicml.utility import DataTuple

__all__ = ["TorchCelebA"]


class TorchCelebA(VisionDataset):
    """Large-scale CelebFaces Attributes (CelebA) Dataset."""

    base_folder = "celeba"

    file_list = [
        (
            "0B7EVK8r0v71pZjFTYXZWM3FlRnM",  # File ID
            "00d2c5bc6d35e252742224ab0c1e8fcb",  # MD5 Hash
            "img_align_celeba.zip",  # Filename
        ),
        (
            "0B7EVK8r0v71pblRyaVFSWGxPY0U",
            "75e246fa4810816ffd6ee81facbd244c",
            "list_attr_celeba.txt",
        ),
        (
            "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
            "d32c9cbf5e040fd4025c592c306e6668",
            "list_eval_partition.txt",
        ),
    ]

    def __init__(
        self,
        data: DataTuple,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """Large-scale CelebFaces Attributes (CelebA) Dataset.

        <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
        Adapted from torchvision.datasets to enable the loading of data triplets and biased/unbiased
        subsets while removing superfluous (for our purposes) elements of the dataset (e.g. facial
        landmarks).

        Args:
            data: A CelebA dataset object.
            root: Root directory where images are downloaded to.
            transform: A function/transform that  takes in an PIL image and returns a transformed
                       version. E.g, `transforms.ToTensor`
            target_transform: A function/transform that takes in the target and transforms it.
            download: If true, downloads the dataset from the internet and puts it in root
                      directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        sens_attr = data.s
        sens_attr = (sens_attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.s_dim = 1

        target_attr = data.y
        target_attr: pd.DataFrame = (target_attr + 1) // 2  # map from {-1, 1} to {0, 1}

        filename = data.x["filename"]

        self.filename: np.ndarray[np.str_] = filename.to_numpy()
        self.sens_attr = torch.as_tensor(sens_attr.to_numpy())
        self.target_attr = torch.as_tensor(target_attr.to_numpy())

    def _check_integrity(self) -> bool:
        """Check integrity of the data folder.

        Returns:
            bool: Boolean indicating whether the file containing the celeba data
                  is readable.
        """
        base = Path(self.root) / self.base_folder
        for (_, md5, filename) in self.file_list:
            fpath = base / filename
            ext = fpath.suffix
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(str(fpath), md5):
                return False

        # Should check a hash of the images
        return (base / "img_align_celeba").is_dir()

    def download(self) -> None:
        """Attempt to download data if files cannot be found in the base folder."""
        import zipfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(
                file_id, os.path.join(self.root, self.base_folder), filename, md5
            )

        with zipfile.ZipFile(
            os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r"
        ) as fhandle:
            fhandle.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Fetch the data sample at the given index.

        Args:
            index (int): Index of the sample to be loaded in.

        Returns:
            Tuple[1]: Tuple containing the sample along
            with its sensitive and target attribute labels.
        """
        x = Image.open(
            os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        )
        s = self.sens_attr[index]
        target = self.target_attr[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x, s, target

    def __len__(self) -> int:
        """Length (sample count) of the dataset.

        Returns:
            Integer indicating the length of the dataset.
        """
        return self.sens_attr.size(0)
