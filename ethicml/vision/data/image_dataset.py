"""Class for loading images.

Modifies the Pytorch vision dataset by enabling the use of sensitive attributes
and biased subset sampling.
"""
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset

from ethicml.data.util import LabelSpec, simple_spec
from ethicml.data.vision_data.celeba import CelebAttrs, celeba
from ethicml.data.vision_data.genfaces import GenfacesAttributes, genfaces
from ethicml.preprocessing import ProportionalSplit, get_biased_subset
from ethicml.utility import DataTuple

__all__ = ["TorchImageDataset", "create_celeba_dataset", "create_genfaces_dataset"]


class TorchImageDataset(VisionDataset):
    """Image dataset for pytorch."""

    def __init__(
        self,
        data: DataTuple,
        root: Path,
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
            transform: A function/transform that  takes in an PIL image and returns a transformed
                       version. E.g, `transforms.ToTensor`
            target_transform: A function/transform that takes in the target and transforms it.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        s = data.s
        self.s_dim = 1
        y = data.y

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


def create_celeba_dataset(
    root: str,
    biased: bool,
    mixing_factor: float,
    unbiased_pcnt: float,
    sens_attr_name: Union[CelebAttrs, Dict[str, List[CelebAttrs]]],
    target_attr_name: CelebAttrs,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    seed: int = 42,
    check_integrity: bool = True,
) -> TorchImageDataset:
    """Create a CelebA dataset object.

    Args:
        root: Root directory where images are downloaded to.
        biased: Wheher to artifically bias the dataset according to the mixing factor. See
                :func:`get_biased_subset()` for more details.
        mixing_factor: Mixing factor used to generate the biased subset of the data.
        unbiased_pcnt: Percentage of the dataset to set aside as the 'unbiased' split.
        sens_attr_name: Attribute(s) to set as the sensitive attribute. Biased sampling cannot be
                        performed if multiple sensitive attributes are specified.
        target_attr_name: Attribute to set as the target attribute.
        transform: A function/transform that  takes in an PIL image and returns a transformed
                   version. E.g, `transforms.ToTensor`
        target_transform: A function/transform that takes in the target and transforms it.
        download: If true, downloads the dataset from the internet and puts it in root
                  directory. If dataset is already downloaded, it is not downloaded again.
        seed: Random seed used to sample biased subset.
        check_integrity: If True, check whether the data has been downloaded correctly.
    """
    sens_attr: Union[CelebAttrs, LabelSpec]
    if isinstance(sens_attr_name, dict):
        sens_attr = dict(simple_spec(sens_attr_name))
    else:
        sens_attr = sens_attr_name
    dataset, base_dir = celeba(
        download_dir=root,
        label=target_attr_name,
        sens_attr=sens_attr,
        download=download,
        check_integrity=check_integrity,
    )
    assert dataset is not None
    all_dt = dataset.load()

    if sens_attr_name == target_attr_name:
        warnings.warn("Same attribute specified for both the sensitive and target attribute.")

    unbiased_dt, biased_dt, _ = ProportionalSplit(unbiased_pcnt, start_seed=seed)(all_dt)

    if biased:
        biased_dt, _ = get_biased_subset(
            data=biased_dt, mixing_factor=mixing_factor, unbiased_pcnt=0, seed=seed
        )
        return TorchImageDataset(
            data=biased_dt, root=base_dir, transform=transform, target_transform=target_transform
        )
    else:
        return TorchImageDataset(
            data=unbiased_dt, root=base_dir, transform=transform, target_transform=target_transform
        )


def create_genfaces_dataset(
    root: str,
    biased: bool,
    mixing_factor: float,
    unbiased_pcnt: float,
    sens_attr_name: GenfacesAttributes,
    target_attr_name: GenfacesAttributes,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    seed: int = 42,
    check_integrity: bool = True,
) -> TorchImageDataset:
    """Create a CelebA dataset object.

    Args:
        root: Root directory where images are downloaded to.
        biased: Wheher to artifically bias the dataset according to the mixing factor. See
                :func:`get_biased_subset()` for more details.
        mixing_factor: Mixing factor used to generate the biased subset of the data.
        sens_attr_name: Attribute(s) to set as the sensitive attribute. Biased sampling cannot be
                        performed if multiple sensitive attributes are specified.
        unbiased_pcnt: Percentage of the dataset to set aside as the 'unbiased' split.
        target_attr_name: Attribute to set as the target attribute.
        transform: A function/transform that  takes in an PIL image and returns a transformed
                   version. E.g, `transforms.ToTensor`
        target_transform: A function/transform that takes in the target and transforms it.
        download: If true, downloads the dataset from the internet and puts it in root
                  directory. If dataset is already downloaded, it is not downloaded again.
        seed: Random seed used to sample biased subset.
        check_integrity: If True, check whether the data has been downloaded correctly.
    """
    sens_attr_name = cast(GenfacesAttributes, sens_attr_name.lower())
    target_attr_name = cast(GenfacesAttributes, target_attr_name.lower())

    dataset, base_dir = genfaces(
        download_dir=root,
        label=target_attr_name,
        sens_attr=sens_attr_name,
        download=download,
        check_integrity=check_integrity,
    )
    assert dataset is not None
    all_dt = dataset.load()

    if sens_attr_name == target_attr_name:
        warnings.warn("Same attribute specified for both the sensitive and target attribute.")

    unbiased_dt, biased_dt, _ = ProportionalSplit(unbiased_pcnt, start_seed=seed)(all_dt)

    if biased:
        biased_dt, _ = get_biased_subset(
            data=biased_dt, mixing_factor=mixing_factor, unbiased_pcnt=0, seed=seed
        )
        return TorchImageDataset(
            data=biased_dt, root=base_dir, transform=transform, target_transform=target_transform
        )
    else:
        return TorchImageDataset(
            data=unbiased_dt, root=base_dir, transform=transform, target_transform=target_transform
        )
