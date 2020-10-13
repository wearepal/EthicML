"""Torch Dataset wrapper for EthicML."""

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ethicml import DataTuple, TestTuple
from ethicml.implementations.pytorch_common import _get_info

if TYPE_CHECKING:
    BaseDataset = Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
else:
    BaseDataset = Dataset


def grouped_features_slices(disc_feature_groups: Dict[str, List[str]]) -> List[slice]:
    """Find slices that correspond to the given feature groups."""
    feature_slices = []
    start_idx = 0
    for group in disc_feature_groups.values():
        len_group = len(list(group))
        indexes = slice(start_idx, start_idx + len_group)
        feature_slices.append(indexes)
        start_idx += len_group

    return feature_slices


class DataTupleDataset(BaseDataset):
    """Wrapper for EthicML datasets."""

    def __init__(
        self,
        dataset: Union[DataTuple, TestTuple],
        disc_feature_groups: Dict[str, List[str]],
        cont_features: Sequence[str],
    ):
        """Create DataTupleDataset."""
        cont_features = [feat for feat in cont_features if feat in dataset.x.columns]

        # order the discrete features as they are in the dictionary `disc_feature_groups`
        discrete_features_in_order: List[str] = []
        for group in disc_feature_groups.values():
            discrete_features_in_order += group

        # get slices that correspond to that ordering
        self.disc_feature_group_slices = grouped_features_slices(disc_feature_groups)

        self.x_disc = dataset.x[discrete_features_in_order].to_numpy(dtype=np.float32)
        self.x_cont = dataset.x[cont_features].to_numpy(dtype=np.float32)
        # self.s = dataset.s.to_numpy(dtype=np.float32)
        # self.y = dataset.y.to_numpy(dtype=np.float32)

        _, self.s, self.num, self.xdim, self.sdim, self.x_names, self.s_names = _get_info(dataset)
        if isinstance(dataset, DataTuple):
            self.y = dataset.y.to_numpy(dtype=np.float32)
            self.ydim = dataset.y.shape[1]
            self.y_names = dataset.y.columns
            self.train = True
        elif isinstance(dataset, TestTuple):
            self.train = False
        else:
            raise NotImplementedError("Should only be a DataTuple or a TestTuple")

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
