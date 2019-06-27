"""
Loads Data from .csv files
"""

from pathlib import Path
from typing import List, Dict
import pandas as pd

from ethicml.utility.data_structures import DataTuple
from .configurable_dataset import ConfigurableDataset
from .dataset import Dataset


def load_data(dataset: Dataset, ordered: bool = False) -> DataTuple:
    """Load dataset from its CSV file

    Args:
        dataset: dataset object
        ordered: if True return features such that discrete come first, then continuous

    Returns:
        DataTuple with dataframes of features, labels and sensitive attributes
    """
    dataframe: pd.DataFrame = pd.read_csv(dataset.filepath)
    assert isinstance(dataframe, pd.DataFrame)

    feature_split = dataset.feature_split if not ordered else dataset.ordered_features

    x_data = dataframe[feature_split["x"]]
    s_data = dataframe[feature_split["s"]]
    y_data = dataframe[feature_split["y"]]

    return DataTuple(x=x_data, s=s_data, y=y_data, name=dataset.name)


def create_data_obj(
    filepath: Path, s_columns: List[str], y_columns: List[str], additional_to_drop=None
) -> Dataset:
    """

    Args:
        filepath:
        s_columns:
        y_columns:
        additional_to_drop:

    Returns:

    """
    if additional_to_drop is None:
        additional_to_drop = []

    conf: ConfigurableDataset = ConfigurableDataset()
    conf.filename = filepath.name
    conf.filepath = filepath

    dataframe: pd.DataFrame = pd.read_csv(filepath)

    columns: List[str] = dataframe.columns.values.tolist()
    for s_col in s_columns:
        columns.remove(s_col)
    for y_col in y_columns:
        columns.remove(y_col)
    for additional in additional_to_drop:
        columns.remove(additional)

    feat_split: Dict[str, List[str]] = {"x": columns, "s": s_columns, "y": y_columns}
    conf.feature_split = feat_split

    return conf
