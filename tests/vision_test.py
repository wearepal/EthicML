"""Test for ethicml.vision."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import ethicml.vision as emvi
from ethicml.data.vision_data.celeba import CelebAttr


@pytest.mark.slow
@pytest.mark.parametrize("transform", [emvi.LdColorizer])
def test_label_dependent_transforms(transform):
    """Test label dependent transforms."""
    data = torch.rand((9, 3, 4, 4))
    labels = torch.randint(low=0, high=10, size=(9,))

    colorizer = emvi.LdColorizer(
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
    """Test celeba."""
    train_set = emvi.create_celeba_dataset(
        root="non-existent",
        biased=True,
        mixing_factor=0.0,
        unbiased_pcnt=0.4,
        sens_attr_name=CelebAttr.Male,
        target_attr_name=CelebAttr.Smiling,
        check_integrity=False,
    )
    test_set = emvi.create_celeba_dataset(
        root="non-existent",
        biased=False,
        mixing_factor=0.0,
        unbiased_pcnt=0.4,
        sens_attr_name=CelebAttr.Male,
        target_attr_name=CelebAttr.Smiling,
        check_integrity=False,
    )

    assert len(train_set) == 52855
    assert len(test_set) == 81039

    assert isinstance(train_set, emvi.TorchImageDataset)
    assert isinstance(test_set, emvi.TorchImageDataset)

    assert train_set.possible_labels is not None
    assert train_set.additional_columns is not None
    assert "Blond_Hair" in train_set.possible_labels
    tmp_smiling = train_set.y
    tmp_blond = (
        torch.as_tensor(train_set.additional_columns["Blond_Hair"].to_numpy()).unsqueeze(-1) + 1
    ) // 2
    assert not torch.equal(tmp_blond, tmp_smiling)

    with pytest.raises(AssertionError):
        train_set.new_task("Blond")
    train_set.new_task("Blond_Hair")
    with pytest.raises(AssertionError):
        assert torch.equal(tmp_smiling, train_set.y)
    assert torch.equal(tmp_blond, train_set.y)

    with pytest.raises(AssertionError):
        train_set.new_sensitive("Smling")
    train_set.new_sensitive("Smiling")
    with pytest.raises(AssertionError):
        assert torch.equal(tmp_blond, train_set.s)
    with pytest.raises(AssertionError):
        assert torch.equal(train_set.s, train_set.y)
    assert torch.equal(tmp_smiling, train_set.s)


def test_celeba_multi_s():
    """Test celeba."""
    data = emvi.create_celeba_dataset(
        root="non-existent",
        biased=False,
        mixing_factor=0.0,
        unbiased_pcnt=1.0,
        sens_attr_name={
            "Hair_Color": [CelebAttr.Black_Hair, CelebAttr.Blond_Hair, CelebAttr.Brown_Hair]
        },
        target_attr_name=CelebAttr.Smiling,
        check_integrity=False,
    )

    assert len(data) == 115_309
    assert data.s.shape[1] == 1
    assert np.unique(data.s.numpy()).tolist() == [0, 1, 2]

    assert isinstance(data, emvi.TorchImageDataset)


def test_gen_faces():
    """Test gen faces."""
    train_set = emvi.create_genfaces_dataset(
        root="non-existent",
        biased=True,
        mixing_factor=0.0,
        unbiased_pcnt=0.4,
        sens_attr_name="gender",
        target_attr_name="emotion",
        check_integrity=False,
    )
    test_set = emvi.create_genfaces_dataset(
        root="non-existent",
        biased=False,
        mixing_factor=0.0,
        unbiased_pcnt=0.4,
        sens_attr_name="gender",
        target_attr_name="emotion",
        check_integrity=False,
    )

    assert len(train_set) == 27928
    assert len(test_set) == 59314

    assert isinstance(train_set, emvi.TorchImageDataset)
    assert isinstance(test_set, emvi.TorchImageDataset)


@pytest.mark.slow
def test_cmnist(temp_dir):
    """Test CMNIST."""
    train_set, test_set = emvi.create_cmnist_datasets(
        root=str(temp_dir), scale=0.01, train_pcnt=0.8, download=True, classes_to_keep=[0, 1]
    )

    train_loader = DataLoader(train_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size=1)

    for _ in range(5):
        x, s, y = next(iter(train_loader))
        assert x.shape == (1, 3, 28, 28)
        assert s.shape == torch.Size([1])
        assert y.shape == torch.Size([1])

        x, s, y = next(iter(test_loader))
        assert x.shape == (1, 3, 28, 28)
        assert s.shape == torch.Size([1])
        assert y.shape == torch.Size([1])
