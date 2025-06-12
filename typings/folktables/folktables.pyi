"""Implements abstract classes for folktables data source and problem definitions."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing_extensions import override

import numpy.typing as npt
import pandas as pd

class DataSource(ABC):
    """Provides access to data source."""

    @abstractmethod
    def get_data(self, **kwargs):  # type: ignore
        """Get data sample from universe.

        Returns:
            Sample.
        """

class Problem(ABC):
    """Abstract class for specifying learning problem."""

    @abstractmethod
    def df_to_numpy(self, df: pd.DataFrame) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Return learning problem as numpy array."""
    # Returns the column name
    @property
    @abstractmethod
    def target(self) -> str:
        pass

    @property
    @abstractmethod
    def features(self):
        pass

    @property
    @abstractmethod
    def target_transform(self):
        pass

class BasicProblem(Problem):
    """Basic prediction or regression problem."""

    def __init__(
        self,
        features: list[str],
        target: str,
        target_transform: Callable[[float], bool] | None = None,
        group: str | None = None,
        group_transform: Callable = lambda x: x,  # type: ignore
        preprocess: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        postprocess: Callable[[npt.NDArray], npt.NDArray] = lambda x: x,
    ):
        """Initialize BasicProblem.

        Args:
            features: list of column names to use as features
            target: column name of target variable
            target_transform: feature transformation for target variable
            group: designated group membership feature
            group_transform: feature transform for group membership
            preprocess: function applied to initial data frame
            postprocess: function applied to final numpy data array
        """

    @override
    def df_to_numpy(self, df: pd.DataFrame) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Return data frame as numpy array.

        Args:
            DataFrame.

        Returns:
            Numpy array, numpy array, numpy array
        """

    def df_to_pandas(
        self,
        df: pd.DataFrame,
        categories: dict[str, dict[float, str]] | None = None,
        dummies: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filters and processes a DataFrame (received from ```ACSDataSource''').

        Args:
            df: pd.DataFrame (received from ```ACSDataSource''')
            categories: nested dict with columns of categorical features
                and their corresponding encodings (see examples folder)
            dummies: bool to indicate the creation of dummy variables for
                categorical features (see examples folder)

        Returns:
            pandas.DataFrame.
        """

    @property
    def target(self) -> str: ...
    @property
    def target_transform(self): ...
    @property
    def features(self) -> list[str]: ...
    @property
    def group(self): ...
    @property
    def group_transform(self): ...
