"""Data source and problem definitions for American Community Survey (ACS) Public Use Microdata Sample (PUMS)."""

from collections.abc import Sequence
from typing import Literal, TypeAlias
from typing_extensions import override

import pandas as pd

from . import folktables
from .load_acs import Horizon, Survey

SurveyYear: TypeAlias = Literal["2014", "2015", "2016", "2017", "2018"]

class ACSDataSource(folktables.DataSource):
    """Data source implementation for ACS PUMS data."""

    def __init__(
        self,
        survey_year: SurveyYear,
        horizon: Horizon,
        survey: Survey,
        root_dir: str = "data",
    ):
        """Create data source around PUMS data for specific year, time horizon, survey type.

        Args:
            survey_year: String. Year of ACS PUMS data, e.g., '2018'
            horizon: String. Must be '1-Year' or '5-Year'
            survey: String. Must be 'person' or 'household'

        Returns:
            ACSDataSource
        """

    @override
    def get_data(  # type: ignore
        self,
        *,
        states: Sequence[str] | None = None,
        density: float = 1.0,
        random_seed: int = 0,
        join_household: bool = False,
        download: bool = False,
    ) -> pd.DataFrame:
        """Get data from given list of states, density, and random seed. Optionally add household features."""

    def get_definitions(self, download: bool = False) -> pd.DataFrame:
        """
        Gets categorical data definitions dataframe.
        Only works for year>=2017 as previous years don't include .csv definition files.
        """

def adult_filter(data: pd.DataFrame) -> pd.DataFrame:
    """Mimic the filters in place for Adult data.

    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """

ACSIncome: folktables.BasicProblem

ACSEmployment: folktables.BasicProblem

ACSHealthInsurance: folktables.BasicProblem

def public_coverage_filter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
    """

ACSPublicCoverage: folktables.BasicProblem

def travel_time_filter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for the employment prediction task
    """

ACSTravelTime: folktables.BasicProblem

ACSMobility: folktables.BasicProblem

def employment_filter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for the employment prediction task
    """

ACSEmploymentFiltered = folktables.BasicProblem

ACSIncomePovertyRatio: folktables.BasicProblem
