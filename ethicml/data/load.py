"""Load Data from .csv files."""
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ethicml.utility import DataTuple

from .dataset import Dataset

__all__ = ["load_data", "create_data_obj"]


def load_data(dataset: Dataset, ordered: bool = False, generate_dummies: bool = False) -> DataTuple:
    """Load dataset from its CSV file.

    This function only exists for backwards compatibility. Use dataset.load() instead.

    Args:
        dataset: dataset object
        ordered: if True, return features such that discrete come first, then continuous
        generate_dummies: if True, generate complementary features for standalone binary features

    Returns:
        DataTuple with dataframes of features, labels and sensitive attributes
    """
    return dataset.load(ordered=ordered, generate_dummies=generate_dummies)


def create_data_obj(
    filepath: Path,
    s_columns: List[str],
    y_columns: List[str],
    additional_to_drop: Optional[List[str]] = None,
) -> Dataset:
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

    dataframe: pd.DataFrame = pd.read_csv(filepath)

    columns: List[str] = [str(x) for x in dataframe.columns.to_numpy().tolist()]
    for s_col in s_columns:
        columns.remove(s_col)
    for y_col in y_columns:
        columns.remove(y_col)
    for additional in additional_to_drop:
        columns.remove(additional)

    return Dataset(
        name=filepath.name,
        num_samples=len(dataframe),
        features=columns,
        cont_features=[],
        sens_attrs=s_columns,
        class_labels=y_columns,
        filename_or_path=filepath,
        discrete_only=False,
    )
