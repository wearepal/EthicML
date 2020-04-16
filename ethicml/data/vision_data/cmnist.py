"""Colourised MNIST dataset.

In the training set the colour is a proxy for the class label,
but at test time the colour is random.
"""

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from ethicml.vision import LdColorizer

from .dataset_wrappers import DatasetWrapper, LdTransformedDataset
from .transforms import NoisyDequantize, Quantize

__all__ = ["create_cmnist_datasets"]


def create_cmnist_datasets(
    *,
    root: str,
    scale: float,
    test_pcnt: float,
    download: bool = False,
    seed: int = 42,
    rotate_data: bool = False,
    shift_data: bool = False,
    padding: bool = False,
    quant_level: int = 8,
    input_noise: bool = False,
) -> Tuple[LdTransformedDataset, LdTransformedDataset]:
    """Create and return colourised MNIST train, test pair.

    Args:
        root: Where the images are downloaded to.
        scale: The amount of 'bias' in the colour. Lower is more biased.
        test_pcnt: The percentage of data to make the test set.
        download: Whether or not to download the data.
        seed: Random seed for reproducing results.
        rotate_data: Whether or not to rotate the training images.
        shift_data: Whether or not to shift the training images.
        padding: Whether or not to pad the training images.
        quant_level: the number of bins to quantize the data into.
        input_noise: Whether or not to add noise to the training images.

    Returns: tuple of train and test data as a Dataset.

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
    all_data: ConcatDataset = ConcatDataset([mnist_train, mnist_test])

    dataset_size = len(all_data)
    indices = list(range(dataset_size))
    split = int(np.floor((1 - test_pcnt) * dataset_size))

    np.random.shuffle(np.array(indices))

    train_indices, test_indices = indices[:split], indices[split:]

    train_data = Subset(all_data, indices=train_indices)
    test_data = Subset(all_data, indices=test_indices)

    colorizer = LdColorizer(
        scale=scale, background=False, black=True, binarize=True, greyscale=False,
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
