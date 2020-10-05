"""Colourised MNIST dataset.

In the training set the colour is a proxy for the class label,
but at test time the colour is random.
"""

import random
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from typing_extensions import Literal

from .dataset_wrappers import DatasetWrapper, LdTransformedDataset
from .label_dependent_transforms import LdColorizer
from .transforms import NoisyDequantize, Quantize
from .utils import train_test_split

__all__ = ["create_cmnist_datasets"]


_Classes = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def _filter_classes(dataset: MNIST, classes_to_keep: Sequence[int]) -> Subset:
    targets: np.ndarray[np.int64] = dataset.targets.numpy()
    final_mask = np.zeros_like(targets, dtype=np.bool_)
    for index, label in enumerate(classes_to_keep):
        mask = targets == label
        targets = np.where(mask, index, targets)
        final_mask |= mask
    dataset.targets = targets
    inds = final_mask.nonzero()[0].tolist()

    return Subset(dataset, inds)


def create_cmnist_datasets(
    *,
    root: str,
    scale: float,
    train_pcnt: float,
    download: bool = False,
    seed: int = 42,
    rotate_data: bool = False,
    shift_data: bool = False,
    padding: bool = False,
    quant_level: int = 8,
    input_noise: bool = False,
    classes_to_keep: Optional[Sequence[_Classes]] = None,
) -> Tuple[LdTransformedDataset, LdTransformedDataset]:
    """Create and return colourised MNIST train/test pair.

    Args:
        root: Where the images are downloaded to.
        scale: The amount of 'bias' in the colour. Lower is more biased.
        train_pcnt: The percentage of data to make the test set.
        download: Whether or not to download the data.
        seed: Random seed for reproducing results.
        rotate_data: Whether or not to rotate the training images.
        shift_data: Whether or not to shift the training images.
        padding: Whether or not to pad the training images.
        quant_level: the number of bins to quantize the data into.
        input_noise: Whether or not to add noise to the training images.
        classes_to_keep: Which digit classes to keep. If None or empty then all classes will be kept.

    Returns:
        tuple of train and test data as a Dataset.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    base_aug = [transforms.ToTensor()]
    data_aug = []
    if rotate_data:
        data_aug.append(transforms.RandomAffine(degrees=15))
    if shift_data:
        data_aug.append(transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)))
    if padding > 0:
        base_aug.insert(0, transforms.Pad(padding))
    if quant_level != 8:
        base_aug.append(Quantize(int(quant_level)))
    if input_noise:
        base_aug.append(NoisyDequantize(int(quant_level)))

    mnist_train = MNIST(root=root, train=True, download=download)
    mnist_test = MNIST(root=root, train=False, download=download)

    if classes_to_keep:
        mnist_train = _filter_classes(dataset=mnist_train, classes_to_keep=classes_to_keep)
        mnist_test = _filter_classes(dataset=mnist_test, classes_to_keep=classes_to_keep)

    all_data: ConcatDataset = ConcatDataset([mnist_train, mnist_test])
    train_data, test_data = train_test_split(all_data, train_pcnt=train_pcnt)

    colorizer = LdColorizer(
        scale=scale, background=False, black=True, binarize=True, greyscale=False
    )
    train_data = DatasetWrapper(train_data, transform=base_aug + data_aug)
    train_data = LdTransformedDataset(
        dataset=train_data,
        ld_transform=colorizer,
        target_dim=10,
        label_independent=False,
        discrete_labels=True,
    )
    test_data = DatasetWrapper(test_data, transform=base_aug)
    test_data = LdTransformedDataset(
        test_data,
        ld_transform=colorizer,
        target_dim=10,
        label_independent=True,
        discrete_labels=True,
    )

    return train_data, test_data
