"""Scale a dataset."""
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from ethicml.data import Dataset
    from ethicml.utility.data_structures import DataTuple

__all__ = ["ScalerType", "scale_continuous"]


class ScalerType(Protocol):
    """Protocol describing a scaler class."""

    def fit(self, df: pd.DataFrame) -> None:
        """Fit parameters of the transformation to the given data."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit parameters of the transformation to the given data and then transform."""

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Invert the transformation."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the given data."""


def scale_continuous(
    dataset: Dataset,
    datatuple: DataTuple,
    scaler: ScalerType,
    inverse: bool = False,
    fit: bool = True,
) -> tuple[DataTuple, ScalerType]:
    """Use a scaler on just the continuous features.

    :example:
        >>> dataset = adult()
        >>> datatuple = dataset.load()
        >>> train, test = train_test_split(datatuple)
        >>> train, scaler = scale_continuous(dataset, train, scaler)
        >>> test, scaler = scale_continuous(dataset, test, scaler, fit=False)

    :param dataset: Dataset object. Used to find the continuous features.
    :param datatuple: DataTuple on which to sclae the continuous features.
    :param scaler: Scaler object to scale the features. Must fit the SKLearn scaler API.
    :param inverse: Should the scaling be reversed? (Default: False)
    :param fit: If not `inverse`, should the scaler be fit to the data? If `True`, do
        `fit_transform` operation, else just `transform`. (Default: True)
    :returns: Tuple of (scaled) DataTuple, and the Scaler (which may have been fit to the data).
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
    return datatuple.replace(x=new_feats), scaler
