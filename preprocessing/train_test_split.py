import pandas as pd
from typing import Tuple


def train_test_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data: pd.DataFrame = data.sample(frac=1, random_state=1)
    assert isinstance(data, pd.DataFrame)
    return data, data
