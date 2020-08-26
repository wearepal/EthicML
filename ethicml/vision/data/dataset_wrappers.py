"""Script containing dataset wrapping classes."""
from typing import Any, Callable, Iterator, Optional, Sequence, Sized, Tuple, TypeVar, Union, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Protocol

__all__ = ["DatasetWrapper", "LdTransformedDataset"]

_T_co = TypeVar("_T_co", covariant=True)


class SizedItemGetter(Sized, Protocol[_T_co]):  # pylint: disable=inherit-non-class
    """A protocol for objects that have a size and implement the __getitem__ method."""

    def __getitem__(self, index: int) -> _T_co:
        ...


TransformType = Union[Callable[..., Tensor], Sequence[Callable]]


class DatasetWrapper(Dataset):
    """Generic dataset wrapper."""

    def __init__(self, dataset: SizedItemGetter, transform: Optional[TransformType] = None):
        self.dataset = dataset
        self.transform = transform

    @property
    def transform(self) -> Optional[TransformType]:
        """The transformation(s) to be applied to the data."""
        return self.__transform

    @transform.setter
    def transform(self, transform: Optional[TransformType]) -> None:
        transform = transform or []
        if not isinstance(transform, Sequence):
            transform = [transform]
        assert transform is not None
        self.__transform = transform

    def _apply_transforms(self, x: Tensor, **kwargs: Any) -> Tensor:
        if self.transform is not None:
            if isinstance(self.transform, Sequence):
                for tform in self.transform:
                    x = tform(x, **kwargs)
            else:
                x = cast(Callable[..., Tensor], self.transform)(x, **kwargs)

        return x

    @staticmethod
    def _validate_data(*args: Union[Tensor, np.ndarray, int, float]) -> Iterator[Tensor]:
        for arg in args:
            if not isinstance(arg, Tensor):
                dtype = torch.long if isinstance(arg, int) else torch.float32
                arg = torch.as_tensor(arg, dtype=dtype)
            if arg.dim() == 0:
                arg = arg.view(-1)
            yield arg

    def __len__(self) -> int:
        """Get the length of the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sequence[Tensor]:
        """Get the item from the dataset at the given index."""
        data = self.dataset[index]
        x = data[0]
        x = self._apply_transforms(x)

        return (x,) + data[1:]


class LdTransformedDataset(DatasetWrapper):
    """Dataset applying label-dependent transformations."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        target_dim: int,
        discrete_labels: bool,
        ld_transform: Optional[TransformType] = None,
        label_independent: bool = False,
        correlation: float = 1.0,
    ):
        super().__init__(dataset=dataset, transform=ld_transform)

        if not 0 <= correlation <= 1:
            raise ValueError("Label correlation must be between 0 and 1.")

        self.target_dim = target_dim
        self.label_independent = label_independent
        self.correlation = correlation
        self.discrete_labels = discrete_labels

    @staticmethod
    def _validate_dataset(dataset: Union[Dataset, DataLoader]) -> Dataset:
        if isinstance(dataset, DataLoader):
            dataset = dataset.dataset
        elif not isinstance(dataset, Dataset):
            raise TypeError("Dataset must be a Dataset or Dataloader object.")

        return dataset

    def _subroutine(self, data: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        x, y = data[0], data[-1]
        s = y
        x, s, y = self._validate_data(x, s, y)

        if self.label_independent:
            if self.discrete_labels:
                s = torch.randint_like(s, low=0, high=self.target_dim)
            else:
                s = torch.rand_like(s)

        if self.correlation < 1:
            flip_prob = torch.rand(s.shape)
            indexes = flip_prob > self.correlation
            s[indexes] = s[indexes][torch.randperm(indexes.size(0))]

        x = self._apply_transforms(x, labels=s)
        x = x.squeeze(0)
        s = s.squeeze(0)
        y = y.squeeze(0)

        return x, s, y

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get an item from the wrapped dataset along with the generated 's' value."""
        return self._subroutine(self.dataset[index])
