"""Torch Dataset wrapper for EthicML."""

from itertools import groupby
from typing import TYPE_CHECKING, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ethicml import DataTuple, TestTuple
from ethicml.implementations.pytorch_common import _get_info

if TYPE_CHECKING:
    BaseDataset = Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
else:
    BaseDataset = Dataset


def group_features(disc_feats: List[str]) -> Iterator[Tuple[str, Iterator[str]]]:
    """Group discrete features names according to the first segment of their name."""

    def _first_segment(feature_name: str) -> str:
        return feature_name.split("_")[0]

    return groupby(disc_feats, _first_segment)


def grouped_features_indexes(disc_feats: List[str]) -> List[slice]:
    """Group discrete features names according to the first segment of their name.

    Then return a list of their corresponding slices (assumes order is maintained).
    """
    group_iter = group_features(disc_feats)

    feature_slices = []
    start_idx = 0
    for _, group in group_iter:
        len_group = len(list(group))
        indexes = slice(start_idx, start_idx + len_group)
        feature_slices.append(indexes)
        start_idx += len_group

    return feature_slices


class DataTupleDataset(BaseDataset):
    """Wrapper for EthicML datasets."""

    def __init__(
        self,
        dataset: DataTuple,
        disc_features: List[str],
        cont_features: List[str],
    ):
        """Create DataTupleDataset."""
        if isinstance(dataset, DataTuple):
            self.train = True
        elif isinstance(dataset, TestTuple):
            self.train = False
        else:
            raise NotImplementedError("Should only be a DataTuple or a TestTuple")

        disc_features = [feat for feat in disc_features if feat in dataset.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in cont_features if feat in dataset.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

        self.x_disc = dataset.x[self.disc_features].to_numpy(dtype=np.float32)
        self.x_cont = dataset.x[self.cont_features].to_numpy(dtype=np.float32)
        # self.s = dataset.s.to_numpy(dtype=np.float32)
        # self.y = dataset.y.to_numpy(dtype=np.float32)

        _, self.s, self.num, self.xdim, self.sdim, self.x_names, self.s_names = _get_info(dataset)
        if self.train:
            self.y = dataset.y.to_numpy(dtype=np.float32)
            self.ydim = dataset.y.shape[1]
            self.y_names = dataset.y.columns

    def __len__(self) -> int:
        return self.s.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_disc = self.x_disc[index]
        x_cont = self.x_cont[index]
        s = self.s[index]
        if self.train:
            y = self.y[index]

        x = np.concatenate([x_disc, x_cont], axis=0)
        x = torch.from_numpy(x).squeeze(0)
        s = torch.from_numpy(s).squeeze()
        if self.train:
            y = torch.from_numpy(y).squeeze()
        else:
            y = torch.zeros_like(s)

        return x, s, y
