"""Test for ethicml.vision."""
import pytest
import torch

from ethicml.data import create_celeba_dataset
from ethicml.vision import LdColorizer, TorchCelebA


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
    _check_integrity = TorchCelebA._check_integrity
    TorchCelebA._check_integrity = lambda _: True  # type: ignore[assignment]

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

    assert len(train_set) == 52725
    assert len(test_set) == 81040

    # restore integrity checking
    TorchCelebA._check_integrity = _check_integrity  # type: ignore[assignment]
