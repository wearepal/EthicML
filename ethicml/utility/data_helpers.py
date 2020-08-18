"""Helper functions for working with DataFrames (and Series)."""
from typing import Optional, Union, overload

import pandas as pd

__all__ = ["undo_one_hot", "shuffle_df"]


def shuffle_df(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Shuffle a given dataframe."""
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


@overload
def undo_one_hot(df: pd.DataFrame, new_column_name: None = None) -> pd.Series:
    ...


@overload
def undo_one_hot(df: pd.DataFrame, new_column_name: str) -> pd.DataFrame:
    ...


def undo_one_hot(
    df: pd.DataFrame, new_column_name: Optional[str] = None
) -> Union[pd.Series, pd.DataFrame]:
    """Undo one-hot encoding."""
    # we have to overwrite the column names because `idxmax` uses the column names
    df.columns = pd.Index(range(df.shape[1]))
    result = df.idxmax(axis="columns")
    if new_column_name is not None:
        return result.to_frame(name=new_column_name)
    else:
        return result
