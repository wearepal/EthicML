"""Label dependent Augmented dataset."""
from typing import Generator, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torchvision.transforms import transforms

from ethicml.data.vision_data.utils import set_transform
from ethicml.vision import LdTransformation


class LdAugmentedDataset(Dataset):
    """Label dependent Augmented dataset."""

    def __init__(
        self,
        source_dataset: Dataset,
        ld_augmentations: Union[LdTransformation, List[LdTransformation]],
        num_classes: int,
        li_augmentation: bool = False,
        base_augmentations: Optional[List[object]] = None,
        correlation: float = 1.0,
    ) -> None:
        """Create a Label Dependent Augmented Dataset.

        Args:
            source_dataset: Dataset to transform.
            ld_augmentations: Label Dependent augmentations.
            num_classes: Number of classes.
            li_augmentation: Label Independent augmentations.
            base_augmentations: Augmentations to be applied to all samples.
            correlation: Probability threshold to do something to the data.... ????
        """
        self.source_dataset = self._validate_dataset(source_dataset)
        if not 0 <= correlation <= 1:
            raise ValueError("Label-augmentation correlation must be between 0 and 1.")

        self.num_classes = num_classes

        if not isinstance(ld_augmentations, (list, tuple)):
            ld_augmentations = [ld_augmentations]
        self.ld_augmentations = ld_augmentations

        if base_augmentations is not None:
            base_augmentations = transforms.Compose(base_augmentations)
            set_transform(self.source_dataset, base_augmentations)
        self.base_augmentations = base_augmentations

        self.li_augmentation = li_augmentation
        self.correlation = correlation

        self.dataset = self._validate_dataset(source_dataset)

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.dataset)

    @staticmethod
    def _validate_dataset(dataset: Union[Dataset, DataLoader]) -> Dataset:
        if isinstance(dataset, DataLoader):
            dataset = dataset
        elif not isinstance(dataset, Dataset):
            raise TypeError("Dataset must be a Dataset or Dataloader object.")

        assert isinstance(dataset, Dataset)
        return dataset

    def subsample(self, pcnt: float = 1.0) -> None:
        """Get a random subsample of the data."""
        if not 0 <= pcnt <= 1.0:
            raise ValueError(f"{pcnt} should be in the range (0, 1]")
        num_samples = int(pcnt * len(self.source_dataset))
        inds = list(RandomSampler(self.source_dataset, num_samples=num_samples, replacement=False))
        self.inds = inds
        subset = self._sample_from_inds(inds)
        self.dataset = subset

    def _sample_from_inds(self, inds: Sequence[int]) -> Subset:
        subset = Subset(self.source_dataset, inds)

        return subset

    @staticmethod
    def _validate_data(
        *args: Union[Tensor, Sequence[int], Sequence[float]]
    ) -> Generator[Tensor, None, None]:
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                dtype = torch.long if type(arg) == int else torch.float32
                arg = torch.tensor(arg, dtype=dtype)
            if arg.dim() == 0:
                arg = arg.view(-1)
            yield (arg)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get Item."""
        return self._subroutine(self.dataset.__getitem__(index))

    def _augment(self, x: Tensor, label: Tensor) -> Tensor:
        for aug in self.ld_augmentations:
            x = aug(x, label)

        return x

    def _subroutine(self, data: DataLoader) -> Tuple[Tensor, Tensor, Tensor]:

        x, y = data
        s = y
        x, s, y = self._validate_data(x, s, y)

        if self.li_augmentation:
            s = torch.randint_like(s, low=0, high=self.num_classes)

        if self.correlation < 1:
            flip_prob = torch.rand(s.shape)
            indexes = flip_prob > self.correlation
            s[indexes] = s[indexes][torch.randperm(indexes.size(0))]

        x = self._augment(x, s)

        if x.dim() == 4 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.squeeze(0)
        s = s.squeeze()
        y = y.squeeze()

        return x, s, y
