"""Scale a dataset."""
from typing import Tuple

import pandas as pd
from typing_extensions import Protocol

from ethicml.data.dataset import Dataset
from ethicml.utility import DataTuple

__all__ = ["scale_continuous"]


class ScalerType(Protocol):
    def fit(self, df: pd.DataFrame) -> None:
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


def scale_continuous(
    dataset: Dataset, datatuple: DataTuple, scaler: ScalerType, inverse: bool = False
) -> Tuple[DataTuple, ScalerType]:
    """Use a scaler on just the continuous features."""
    new_feats = datatuple.x.copy().astype('float64')
    if inverse:
        new_feats[dataset.continuous_features] = scaler.inverse_transform(
            new_feats[dataset.continuous_features]
        )
    else:
        new_feats[dataset.continuous_features] = scaler.fit_transform(
            new_feats[dataset.continuous_features]
        )
    return DataTuple(x=new_feats, s=datatuple.s, y=datatuple.y), scaler
