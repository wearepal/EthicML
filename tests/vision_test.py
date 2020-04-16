"""Test for ethicml.vision."""

import pytest
import torch
from torch.utils.data import DataLoader

from ethicml.data import (
    CelebA,
    GenFaces,
    create_celeba_dataset,
    create_cmnist_datasets,
    create_genfaces_dataset,
)
from ethicml.vision import LdColorizer, TorchImageDataset


@pytest.mark.parametrize("transform", [LdColorizer])
def test_label_dependent_transforms(transform):
    """test label dependent transforms"""

    data = torch.rand((9, 3, 4, 4))
    labels = torch.randint(low=0, high=10, size=(9,))

    colorizer = LdColorizer(
        scale=0.02,
        min_val=0.0,
        max_val=1.0,
        binarize=False,
        background=False,
        black=False,
        seed=47,
        greyscale=False,
    )

    colorizer(data, labels)


def test_celeba():
    """test celeba"""
    # pylint: disable=protected-access

    # turn off integrity checking for testing purposes
    check_integrity = CelebA.check_integrity
    CelebA.check_integrity = lambda _: True  # type: ignore[assignment]

    train_set = create_celeba_dataset(
        root="non-existent",
        biased=True,
        mixing_factor=0.0,
        unbiased_pcnt=0.4,
        sens_attr_name="Male",
        target_attr_name="Smiling",
    )
    test_set = create_celeba_dataset(
        root="non-existent",
        biased=False,
        mixing_factor=0.0,
        unbiased_pcnt=0.4,
        sens_attr_name="Male",
        target_attr_name="Smiling",
    )

    assert len(train_set) == 52855
    assert len(test_set) == 81039

    assert isinstance(train_set, TorchImageDataset)
    assert isinstance(test_set, TorchImageDataset)

    # restore integrity checking
    CelebA.check_integrity = check_integrity  # type: ignore[assignment]


def test_gen_faces():
    """test gen faces"""
    # pylint: disable=protected-access

    # turn off integrity checking for testing purposes
    check_integrity = GenFaces.check_integrity
    GenFaces.check_integrity = lambda _: True  # type: ignore[assignment]

    train_set = create_genfaces_dataset(
        root="non-existent",
        biased=True,
        mixing_factor=0.0,
        unbiased_pcnt=0.4,
        sens_attr_name="gender",
        target_attr_name="emotion",
    )
    test_set = create_genfaces_dataset(
        root="non-existent",
        biased=False,
        mixing_factor=0.0,
        unbiased_pcnt=0.4,
        sens_attr_name="gender",
        target_attr_name="emotion",
    )

    assert len(train_set) == 27928
    assert len(test_set) == 59314

    assert isinstance(train_set, TorchImageDataset)
    assert isinstance(test_set, TorchImageDataset)

    # restore integrity checking
    GenFaces.check_integrity = check_integrity  # type: ignore[assignment]


def test_cmnist(temp_dir):
    train_set, test_set = create_cmnist_datasets(
        root=str(temp_dir), scale=0.01, test_pcnt=0.2, download=True, labels_to_keep=[0, 1]
    )

    train_loader = DataLoader(train_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size=1)

    for i in range(5):
        x, s, y = next(iter(train_loader))
        assert x.shape == (1, 3, 28, 28)
        assert s.shape == torch.Size([1])
        assert y.shape == torch.Size([1])

        x, s, y = next(iter(test_loader))
        assert x.shape == (1, 3, 28, 28)
        assert s.shape == torch.Size([1])
        assert y.shape == torch.Size([1])
