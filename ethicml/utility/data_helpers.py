"""Helper functions for working with DataFrames (and Series)."""
import pandas as pd


def shuffle_df(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Shuffle a given dataframe."""
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
