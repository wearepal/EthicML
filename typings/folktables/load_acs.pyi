"""Load ACS PUMS data from Census CSV files."""

from typing import Final, Literal, TypeAlias

import pandas as pd

state_list: Final[list[str]]

_STATE_CODES: Final[dict[str, str]]

Horizon: TypeAlias = Literal["1-Year", "5-Year"]
Survey: TypeAlias = Literal["person", "household"]

def download_and_extract(
    url: str,
    datadir: str,
    remote_fname: str,
    file_name: str,
    delete_download: bool = False,
):
    """Helper function to download and unzip files."""

def initialize_and_download(
    datadir: str,
    state: str,
    year: str | int,
    horizon: Horizon,
    survey: Survey,
    download: bool = False,
) -> str:
    """Download the dataset (if required)."""

def load_acs(
    root_dir: str,
    states: list[str] | None = None,
    year: str | int = 2018,
    horizon: Horizon = "1-Year",
    survey: Survey = "person",
    density: int = 1,
    random_seed: int = 1,
    serial_filter_list: list[int] | None = None,
    download: bool = False,
) -> pd.DataFrame:
    """
    Load sample of ACS PUMS data from Census csv files into DataFrame.

    If a serial filter list is passed in, density and random_seed are ignored
    and the output is instead filtered with the provided list (only entries with
    a serial number in the list are kept).
    """

def load_definitions(
    root_dir: str,
    year: str | int = 2018,
    horizon: Horizon = "1-Year",
    download: bool = False,
) -> pd.DataFrame:
    """
    Loads the data attribute definition file.

    File only available for year >= 2017.
    """

def generate_categories(
    features: list[str], definition_df: pd.DataFrame
) -> dict[str, dict[float, str]]:
    """Generates a categories dictionary using the provided definition dataframe. Does not create a category mapping
    for variables requiring the 2010 Public use microdata area code (PUMA) as these need an additional definition
    file which are not unique without the state code.

    Args:
        features: list (list of features to include in the categories dictionary, numeric features will be ignored)
        definition_df: pd.DataFrame (received from ```ACSDataSource.get_definitions()''')

    Returns:
        categories: nested dict with columns of categorical features
            and their corresponding encodings (see examples folder)."""
