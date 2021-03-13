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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


def scale_continuous(
    dataset: Dataset,
    datatuple: DataTuple,
    scaler: ScalerType,
    inverse: bool = False,
    fit: bool = True,
) -> Tuple[DataTuple, ScalerType]:
    """Use a scaler on just the continuous features.

    Args:
        dataset:
            Dataset object. Used to find the continuous features.
        datatuple:
            DataTuple on which to sclae the continuous features.
        scaler:
            Scaler object to scale the features. Must fit the SKLearn scaler API.
        inverse:
            Should the scaling be reversed?
        fit:
            If not `inverse`, should the scaler be fit to the data? If `True`, do
            `fit_transform` operation, else just `transform`.

    Returns:
        Tuple of (scaled) DataTuple, and the Scaler (which may have been fit to the data).

    Examples:
        >>> dataset = adult()
        >>> datatuple = dataset.load()
        >>> train, test = train_test_split(datatuple)
        >>> train, scaler = scale_continuous(dataset, train, scaler)
        >>> test, scaler = scale_continuous(dataset, test, scaler, fit=False)
    """
    new_feats = datatuple.x.copy().astype('float64')
    if inverse:
        new_feats[dataset.continuous_features] = scaler.inverse_transform(
            new_feats[dataset.continuous_features]
        )
    elif fit:
        new_feats[dataset.continuous_features] = scaler.fit_transform(
            new_feats[dataset.continuous_features]
        )
    else:
        new_feats[dataset.continuous_features] = scaler.transform(
            new_feats[dataset.continuous_features]
        )
    return DataTuple(x=new_feats, s=datatuple.s, y=datatuple.y), scaler
