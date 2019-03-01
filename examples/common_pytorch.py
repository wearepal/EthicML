"""Functions that are common to models that use PyTorch"""
import torch
from torch.utils.data import TensorDataset

from .common import DataTuple


def to_pytorch_dataset(data_tuple: DataTuple):
    """Construct a PyTorch dataset from a DataTuple"""
    x = torch.tensor(data_tuple.x.values, dtype=torch.float32)
    s = torch.tensor(data_tuple.s.values, dtype=torch.float32)
    y = torch.tensor(data_tuple.y.values, dtype=torch.float32)
    return TensorDataset(x, s, y)
