"""
Returns a subset of the data. Used primarily in testing so that kernel methods finish in a
reasonable time
"""
from pathlib import Path
from typing import NamedTuple, Dict, Tuple, List

import pandas as pd


class DataTuple(NamedTuple):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels"""

    x: pd.DataFrame  # features
    s: pd.DataFrame  # senstitive attributes
    y: pd.DataFrame  # class labels


class PathTuple(NamedTuple):
    """For algorithms that run in their own process, we pass around paths to the data"""

    x: Path  # path to file with features
    s: Path  # path to file with sensitive attributes
    y: Path  # path to file with class labels


def write_as_feather(
    train: DataTuple, test: DataTuple, data_dir: Path
) -> (Tuple[PathTuple, PathTuple]):
    """Write the given DataTuple to Feather files and return the file paths as PathTuples

    Args:
        train: tuple with training data
        test: tuple with test data
        data_dir: directory where the files should be stored
    Returns:
        tuple of tuple of paths (one tuple for training, one for test)
    """
    # create the directory if it doesn't already exist
    data_dir.mkdir(parents=True, exist_ok=True)

    train_paths: Dict[str, Path] = {}
    test_paths: Dict[str, Path] = {}
    for data_tuple, data_paths, prefix in [
        (train, train_paths, "train"),
        (test, test_paths, "test"),
    ]:
        # loop over all elements of the data tuple and write them to separate files
        for key, data in data_tuple._asdict().items():
            # SUGGESTION: maybe the file names should be completely random to avoid collisions
            data_path = data_dir / Path(f"data_{prefix}_{key}.feather")
            # write the file (don't use compression because this requires an additional library)
            data.to_feather(data_path)
            data_paths[key] = data_path
    # the paths dictionaries to construct path tuples and return them
    return PathTuple(**train_paths), PathTuple(**test_paths)


def apply_to_joined_tuple(mapper, datatup: DataTuple) -> DataTuple:
    """Concatenate the dataframes in a DataTuple and apply a function to it"""
    cols_x = datatup.x.columns
    cols_s = datatup.s.columns
    cols_y = datatup.y.columns
    joined = pd.concat([datatup.x, datatup.s, datatup.y], axis='columns', sort=False)
    joined = mapper(joined)
    return DataTuple(x=joined[cols_x], s=joined[cols_s], y=joined[cols_y])


def concat_dt(datatup_list: List[DataTuple], axis: str = 'index', ignore_index: bool = False):
    """Concatenate the data tuples in the given list"""
    return DataTuple(
        *[
            pd.concat(
                [datatup[i] for datatup in datatup_list],
                axis=axis,
                sort=False,
                ignore_index=ignore_index,
            )
            for i, _ in enumerate(datatup_list[0])
        ]
    )


def load_feather(output_path: Path) -> pd.DataFrame:
    """Load a dataframe from a feather file"""
    with output_path.open('rb') as file_obj:
        df = pd.read_feather(file_obj)
    return df


def get_subset(train: DataTuple, num: int = 500) -> DataTuple:
    """Get the first elements of the given dataset

    Args:
        train: training data

    Returns:
        subset of training data
    """
    return DataTuple(x=train.x.iloc[:num], s=train.s.iloc[:num], y=train.y.iloc[:num])
