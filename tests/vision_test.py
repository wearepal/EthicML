
import pytest

import torch
from torch.utils.data import DataLoader, random_split

from ethicml.vision import LdColorizer


@pytest.mark.parametrize("transform", [LdColorizer])
def test_label_dependent_transforms(transform):

    data = torch.rand(0, 1, (9, 3, 4, 4))
    labels = torch.randint(low=0, high=10, size=(9,))

    colorizer = LdColorizer(
        min_val=0.,
        max_val=1.,
        scale=0.02,
        binarize=False,
        background=False,
        black=False,
        seed=47,
        greyscale=False
    )

    augmented_data = colorizer(data, labels)
