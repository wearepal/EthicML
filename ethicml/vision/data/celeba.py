"""Class for loading CelebA.

Modifies the Pytorch CelebA dataset by enabling the use of sensitive attributes
and biased subset sampling.
"""
import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Sequence
from typing_extensions import Literal

import pandas as pd
from PIL import Image

import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive

from ethicml.preprocessing import get_biased_subset, SequentialSplit
from ethicml.utility import DataTuple

__all__ = ["CelebA"]


_CELEBATTRS = Literal[
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


class CelebA(VisionDataset):
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
        root: str,
        biased: bool,
        mixing_factor: float,
        unbiased_pcnt: float,
        sens_attrs: Sequence[_CELEBATTRS],
        target_attr_name: _CELEBATTRS,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        seed: int = 42,
    ):
        """Large-scale CelebFaces Attributes (CelebA) Dataset.

        <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
        Adapted from torchvision.datasets to enable the loading of data triplets and biased/unbiased
        subsets while removing superfluous (for our purposes) elements of the dataset (e.g. facial
        landmarks).

        Args:
            root: Root directory where images are downloaded to.
            biased: Wheher to artifically bias the dataset according to the mixing factor. See
                    :func:`get_biased_subset()` for more details.
            mixing_factor: Mixing factor used to generate the biased subset of the data.
            sens_attrs: Attribute(s) to set as the sensitive attribute. Biased sampling cannot be
                        performed if multiple sensitive attributes are specified.
            unbiased_pcnt: Percentage of the dataset to set aside as the 'unbiased' split.
            target_attr_name: Attribute to set as the target attribute.
            transform: A function/transform that  takes in an PIL image and returns a transformed
                       version. E.g, `transforms.ToTensor`
            target_transform: A function/transform that takes in the target and transforms it.
            download: If true, downloads the dataset from the internet and puts it in root
                      directory. If dataset is already downloaded, it is not downloaded again.
            seed: Random seed used to sample biased subset.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        base = Path(self.root) / self.base_folder
        partition_file = base / "list_eval_partition.txt"
        # partition: information about which samples belong to train, val or test
        partition = pd.read_csv(
            partition_file, delim_whitespace=True, header=None, index_col=0, names=["partition"]
        )
        # attrs: all attributes with filenames as index
        attrs = pd.read_csv(base / "list_attr_celeba.txt", delim_whitespace=True, header=1)
        all_data = pd.concat([partition, attrs], axis="columns", sort=False)
        # the filenames are used for indexing; here we turn them into a regular column
        all_data = all_data.reset_index(drop=False).rename(columns={"index": "filenames"})

        attr_names = list(attrs.columns)
        sens_names: List[str] = list(map(str, sens_attrs))

        # Multiple attributes have been designated as sensitive
        # Note that in this case biased dataset sampling cannot be performed
        if len(sens_attrs) > 1:
            if any(sens_attr_name not in attr_names for sens_attr_name in sens_attrs):
                raise ValueError(f"at least one of {sens_attrs} does not exist as an attribute.")
            # only use those samples where exactly one of the specified attributes is true
            all_data = all_data.loc[((all_data[sens_names] + 1) // 2).sum(axis="columns") == 1]
            self.s_dim = len(sens_attrs)
            # perform the reverse operation of one-hot encoding
            data_only_sens = pd.DataFrame(all_data[sens_names], columns=list(range(self.s_dim)))
            sens_attr = data_only_sens.idxmax(axis="columns").to_frame(name=",".join(sens_attrs))
        else:
            sens_attr_name = sens_attrs[0].capitalize()
            if sens_attr_name not in attr_names:
                raise ValueError(f"{sens_attr_name} does not exist as an attribute.")
            sens_attr = all_data[[sens_attr_name]]
            sens_attr = (sens_attr + 1) // 2  # map from {-1, 1} to {0, 1}
            self.s_dim = 1

        target_attr_name = target_attr_name.capitalize()
        if target_attr_name not in attr_names:
            raise ValueError(f"{target_attr_name} does not exist as an attribute.")

        if sens_attr_name == target_attr_name:
            raise ValueError(f"{sens_attr_name} does not exist as an attribute.")

        if sens_attr_name == target_attr_name:
            warnings.warn("Same attribute specified for both the sensitive and target attribute.")

        target_attr = all_data[[target_attr_name]]
        target_attr: pd.DataFrame = (target_attr + 1) // 2  # map from {-1, 1} to {0, 1}

        filename = all_data[["filenames"]]

        all_dt = DataTuple(x=filename, s=sens_attr, y=target_attr)

        # NOTE: the sequential split does not shuffle
        unbiased_dt, biased_dt, _ = SequentialSplit(train_percentage=unbiased_pcnt)(all_dt)

        if biased:
            if self.s_dim == 1:  # FIXME: biasing the dataset only works with binary s right now
                biased_dt, _ = get_biased_subset(
                    data=biased_dt, mixing_factor=mixing_factor, unbiased_pcnt=0, seed=seed
                )
            filename, sens_attr, target_attr = biased_dt
        else:
            filename, sens_attr, target_attr = unbiased_dt

        self.filename = filename.to_numpy()[:, 0]
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
