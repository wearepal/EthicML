"""Load Data from .csv files."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ethicml.utility import DataTuple

from .dataset import Dataset, LegacyDataset

__all__ = ["load_data", "create_data_obj"]


def load_data(dataset: Dataset) -> DataTuple:
    """Load dataset from its CSV file.

    This function only exists for backwards compatibility. Use dataset.load() instead.

    :param dataset: dataset object
    :returns: DataTuple with dataframes of features, labels and sensitive attributes
    """
    return dataset.load()


def create_data_obj(
    filepath: Path, s_column: str, y_column: str, additional_to_drop: list[str] | None = None
) -> ConfigurableDataset:
    """Create a `ConfigurableDataset` from the given file.

    :param filepath: path to a CSV file
    :param s_column: column that represents sensitive attributes
    :param y_column: column that contains lables
    :param additional_to_drop: other columns that should be dropped (Default: None)
    :returns: Dataset object
    """
    return ConfigurableDataset(
        filepath_=filepath,
        s_column=s_column,
        y_column=y_column,
        additional_to_drop=additional_to_drop,
    )


@dataclass
class ConfigurableDataset(LegacyDataset):
    """A configurable dataset class."""

    filepath_: Optional[Path] = None
    s_column: Optional[str] = None
    y_column: Optional[str] = None
    additional_to_drop: Optional[List[str]] = None

    def __post_init__(self) -> None:
        assert self.filepath_ is not None
        assert self.s_column is not None
        assert self.y_column is not None
        if self.additional_to_drop is None:
            self.additional_to_drop = []

        dataframe: pd.DataFrame = pd.read_csv(self.filepath_)

        columns: list[str] = [str(x) for x in dataframe.columns.to_numpy().tolist()]
        columns.remove(self.s_column)
        columns.remove(self.y_column)
        for additional in self.additional_to_drop:
            columns.remove(additional)

        super().__init__(
            name=self.filepath_.name,
            num_samples=len(dataframe),
            features=columns,
            cont_features=[],
            sens_attr_spec=self.s_column,
            class_label_spec=self.y_column,
            filename_or_path=self.filepath_,
        )
