"""Loads Data from .csv files."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ethicml.utility import DataTuple

from .configurable_dataset import ConfigurableDataset
from .dataset import Dataset

__all__ = ["load_data", "create_data_obj"]


def load_data(dataset: Dataset, ordered: bool = False, generate_dummies: bool = True) -> DataTuple:
    """Load dataset from its CSV file.

    Args:
        dataset: dataset object
        ordered: if True, return features such that discrete come first, then continuous
        generate_dummies: if True, generate complementary features for standalone binary features

    Returns:
        DataTuple with dataframes of features, labels and sensitive attributes
    """
    dataframe: pd.DataFrame = pd.read_csv(dataset.filepath)
    assert isinstance(dataframe, pd.DataFrame)

    feature_split = dataset.feature_split if not ordered else dataset.ordered_features

    x_data = dataframe[feature_split["x"]]
    s_data = dataframe[feature_split["s"]]
    y_data = dataframe[feature_split["y"]]

    if generate_dummies:
        # check whether we have to generate some complementary columns for binary features
        disc_feature_groups = dataset.disc_feature_groups
        if disc_feature_groups is not None:
            for group in disc_feature_groups.values():
                if len(group) == 2:
                    # check if one of the features is a dummy feature (not in the CSV file)
                    if group[0] not in x_data.columns:
                        existing_feature, non_existing_feature = group[1], group[0]
                    elif group[1] not in x_data.columns:
                        existing_feature, non_existing_feature = group[0], group[1]
                    else:
                        continue  # all features are present
                    # the dummy feature is the inverse of the existing feature
                    x_data = x_data.assign(**{non_existing_feature: 1 - x_data[existing_feature]})

    return DataTuple(x=x_data, s=s_data, y=y_data, name=dataset.name)


def create_data_obj(
    filepath: Path,
    s_columns: List[str],
    y_columns: List[str],
    additional_to_drop: Optional[List[str]] = None,
) -> ConfigurableDataset:
    """Create a `ConfigurableDataset` from the given file.

    Args:
        filepath: path to a CSV file
        s_columns: list of columns that represent sensitive attributes
        y_columns: list of columns that contain lables
        additional_to_drop: other columns that should be dropped

    Returns:
        Dataset object
    """
    if additional_to_drop is None:
        additional_to_drop = []

    conf: ConfigurableDataset = ConfigurableDataset()
    conf.filename = filepath.name
    conf.filepath = filepath

    dataframe: pd.DataFrame = pd.read_csv(filepath)

    columns: List[str] = [str(x) for x in dataframe.columns.to_numpy().tolist()]
    for s_col in s_columns:
        columns.remove(s_col)
    for y_col in y_columns:
        columns.remove(y_col)
    for additional in additional_to_drop:
        columns.remove(additional)

    feat_split: Dict[str, List[str]] = {"x": columns, "s": s_columns, "y": y_columns}
    conf.feature_split = feat_split

    return conf
