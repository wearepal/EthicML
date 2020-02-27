"""Class for loading GenFaces.

"""
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive

from ethicml.utility import DataTuple

__all__ = ["TorchGenFaces"]


class TorchGenFaces(VisionDataset):
    """PyTorch Dataset for the AI Generated Faces Dataset."""

    base_folder = "genfaces"
    file_id = "1rfwiDmsw37IDnMSWKx5gTc_CZ_Dh5d5g"
    filename = "genfaces_info"

    def __init__(
        self,
        data: DataTuple,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """Dataset of AI-generated Faces.

        <https://generated.photos/faces>

        Args:
            data: A GenFaces dataset object.
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
        target_attr = data.y

        filename = data.x["id"]

        self.filename: np.ndarray[np.str_] = filename.to_numpy()
        self.sens_attr = torch.as_tensor(sens_attr.to_numpy())
        self.target_attr = torch.as_tensor(target_attr.to_numpy())

        self._base = Path(self.root) / self.base_folder

    def _check_integrity(self) -> bool:
        """Check integrity of the data folder.

        Returns:
            bool: Boolean indicating whether the file containing the celeba data
                  is readable.
        """
        fpath = self._base / self.filename
        ext = fpath.suffix
        # Allow original archive to be deleted (zip and 7z)
        # Only need the extracted images
        if ext not in [".zip", ".7z"] and not check_integrity(str(fpath)):
            return False

        return (self._base / "images").is_dir()

    def download(self) -> None:
        """Attempt to download data if files cannot be found in the base folder."""
        import zipfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        fpath = self._base / self.filename
        download_file_from_google_drive(fpath)

        with zipfile.ZipFile(fpath, "r") as fhandle:
            fhandle.extractall(self._base)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Fetch the data sample at the given index.

        Args:
            index (int): Index of the sample to be loaded in.

        Returns:
            Tuple[1]: Tuple containing the sample along
            with its sensitive and target attribute labels.
        """
        x = Image.open(self._base / "images", self.filename[index])
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
