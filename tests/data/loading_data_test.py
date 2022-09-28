"""Test the loading data capability."""
from pathlib import Path
from typing import Callable, Final, Mapping, NamedTuple

import pandas as pd
import pytest

import ethicml as em
from ethicml import DataTuple
from ethicml.data import (
    AcsEmployment,
    AcsIncome,
    Admissions,
    Adult,
    Compas,
    Credit,
    Crime,
    Dataset,
    German,
    Health,
    LabelGroup,
    Law,
    LegacyDataset,
    Lipton,
    NonBinaryToy,
    Nursery,
    Sqf,
    Synthetic,
    SyntheticScenarios,
    SyntheticTargets,
    Toy,
    create_data_obj,
    flatten_dict,
    group_disc_feat_indices,
    spec_from_binary_cols,
)


def test_can_load_test_data(data_root: Path):
    """Test whether we can load test data."""
    data_loc = data_root / "toy.csv"
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


class DT(NamedTuple):
    """Describe data for the tests."""

    dataset: Dataset
    samples: int
    x_features: int
    discrete_features: int
    s_features: int
    num_sens: int
    y_features: int
    num_labels: int
    name: str
    sum_s: int
    sum_y: int

    def __repr__(self):
        return f"{self.name}"


def idfn(val: DT):
    """Use the NamedTuple repr as the pytest ID."""
    return f"{val}"


dts: Final = [
    DT(
        dataset=Admissions(),
        samples=43_303,
        x_features=9,
        discrete_features=0,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Admissions Gender",
        sum_s=22_335,
        sum_y=20_263,
    ),
    DT(
        dataset=Admissions(split=Admissions.Splits.GENDER, invert_s=True),
        samples=43_303,
        x_features=9,
        discrete_features=0,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Admissions Gender",
        sum_s=43_303 - 22_335,
        sum_y=20_263,
    ),
    DT(
        dataset=Adult(),
        samples=45_222,
        x_features=101,
        discrete_features=96,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Adult Sex",
        sum_s=30_527,
        sum_y=11_208,
    ),
    DT(
        dataset=Adult(split=Adult.Splits.SEX, binarize_nationality=True),
        samples=45_222,
        x_features=62,
        discrete_features=57,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Adult Sex, binary nationality",
        sum_s=30_527,
        sum_y=11_208,
    ),
    DT(
        dataset=Adult(split=Adult.Splits.SEX, binarize_race=True),
        samples=45_222,
        x_features=98,
        discrete_features=93,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Adult Sex, binary race",
        sum_s=30_527,
        sum_y=11_208,
    ),
    DT(
        dataset=Adult(split=Adult.Splits.SEX, binarize_nationality=True, binarize_race=True),
        samples=45_222,
        x_features=59,
        discrete_features=54,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Adult Sex, binary nationality, binary race",
        sum_s=30_527,
        sum_y=11_208,
    ),
    DT(
        dataset=Adult(split=Adult.Splits.SEX),
        samples=45_222,
        x_features=101,
        discrete_features=96,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Adult Sex",
        sum_s=30_527,
        sum_y=11_208,
    ),
    DT(
        dataset=Adult(split=Adult.Splits.RACE),
        samples=45_222,
        x_features=98,
        discrete_features=93,
        s_features=1,
        num_sens=5,
        y_features=1,
        num_labels=2,
        name="Adult Race",
        sum_s=166_430,
        sum_y=11_208,
    ),
    DT(
        dataset=Adult(split=Adult.Splits.RACE),
        samples=45_222,
        x_features=98,
        discrete_features=93,
        s_features=1,
        num_sens=5,
        y_features=1,
        num_labels=2,
        name="Adult Race",
        sum_s=166_430,
        sum_y=11_208,
    ),
    DT(
        dataset=Adult(split=Adult.Splits.RACE_BINARY),
        samples=45_222,
        x_features=98,
        discrete_features=93,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Adult Race-Binary",
        sum_s=38_903,
        sum_y=11_208,
    ),
    DT(
        dataset=Adult(split=Adult.Splits.NATIONALITY),
        samples=45_222,
        x_features=62,
        discrete_features=57,
        s_features=1,
        num_sens=41,
        y_features=1,
        num_labels=2,
        name="Adult Nationality",
        sum_s=1_646_127,
        sum_y=11_208,
    ),
    DT(
        dataset=Adult(split=Adult.Splits.EDUCTAION),
        samples=45_222,
        x_features=86,
        discrete_features=82,
        s_features=1,
        num_sens=3,
        y_features=1,
        num_labels=2,
        name="Adult Education",
        sum_s=50_979,
        sum_y=11_208,
    ),
    DT(
        dataset=Compas(),
        samples=6_167,
        x_features=400,
        discrete_features=395,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Compas Sex",
        sum_s=4_994,
        sum_y=2_809,
    ),
    DT(
        dataset=Compas(split=Compas.Splits.SEX),
        samples=6_167,
        x_features=400,
        discrete_features=395,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Compas Sex",
        sum_s=4_994,
        sum_y=2_809,
    ),
    DT(
        dataset=Compas(split=Compas.Splits.RACE),
        samples=6_167,
        x_features=400,
        discrete_features=395,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Compas Race",
        sum_s=2_100,
        sum_y=2_809,
    ),
    DT(
        dataset=Compas(split=Compas.Splits.RACE_SEX),
        samples=6_167,
        x_features=399,
        discrete_features=394,
        s_features=1,
        num_sens=4,
        y_features=1,
        num_labels=2,
        name="Compas Race-Sex",
        sum_s=9_194,
        sum_y=2_809,
    ),
    DT(
        dataset=Credit(),
        samples=30_000,
        x_features=29,
        discrete_features=9,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Credit Sex",
        sum_s=18_112,
        sum_y=6_636,
    ),
    DT(
        dataset=Credit(split=Credit.Splits.SEX),
        samples=30_000,
        x_features=29,
        discrete_features=9,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Credit Sex",
        sum_s=18_112,
        sum_y=6_636,
    ),
    DT(
        dataset=Crime(),
        samples=1_993,
        x_features=136,
        discrete_features=46,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Crime Race-Binary",
        sum_s=970,
        sum_y=653,
    ),
    DT(
        dataset=Crime(split=Crime.Splits.RACE_BINARY),
        samples=1_993,
        x_features=136,
        discrete_features=46,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Crime Race-Binary",
        sum_s=970,
        sum_y=653,
    ),
    DT(
        dataset=German(),
        samples=1_000,
        x_features=57,
        discrete_features=51,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="German Sex",
        sum_s=690,
        sum_y=300,
    ),
    DT(
        dataset=German(split=German.Splits.SEX),
        samples=1_000,
        x_features=57,
        discrete_features=51,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="German Sex",
        sum_s=690,
        sum_y=300,
    ),
    DT(
        dataset=Health(),
        samples=171_067,
        x_features=130,
        discrete_features=12,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Health",
        sum_s=76_450,
        sum_y=54_052,
    ),
    DT(
        dataset=Health(split=Health.Splits.SEX),
        samples=171_067,
        x_features=130,
        discrete_features=12,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Health",
        sum_s=76_450,
        sum_y=54_052,
    ),
    DT(
        dataset=Law(split=Law.Splits.SEX),
        samples=21_791,
        x_features=3,
        discrete_features=0,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Law Sex",
        sum_s=9_537,
        sum_y=19_360,
    ),
    DT(
        dataset=Law(split=Law.Splits.RACE),
        samples=21_791,
        x_features=3,
        discrete_features=0,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Law Race",
        sum_s=18_285,
        sum_y=19_360,
    ),
    DT(
        dataset=Law(split=Law.Splits.SEX_RACE),
        samples=21_791,
        x_features=3,
        discrete_features=0,
        s_features=1,
        num_sens=16,
        y_features=1,
        num_labels=2,
        name="Law Sex-Race",
        sum_s=282_635,
        sum_y=19_360,
    ),
    DT(
        dataset=Lipton(),
        samples=2_000,
        x_features=2,
        discrete_features=0,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Lipton",
        sum_s=989,
        sum_y=-562,
    ),
    DT(
        dataset=NonBinaryToy(),
        samples=400,
        x_features=10,
        discrete_features=8,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=5,
        name="NonBinaryToy",
        sum_s=200,
        sum_y=826,
    ),
    DT(
        dataset=Nursery(),
        samples=12960,
        x_features=22,
        discrete_features=21,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Nursery Finance",
        sum_s=6480,
        sum_y=4320,
    ),
    DT(
        dataset=Sqf(),
        samples=12_347,
        x_features=145,
        discrete_features=139,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="SQF Sex",
        sum_s=11_394,
        sum_y=1_289,
    ),
    DT(
        dataset=Sqf(split=Sqf.Splits.SEX),
        samples=12_347,
        x_features=145,
        discrete_features=139,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="SQF Sex",
        sum_s=11_394,
        sum_y=1_289,
    ),
    DT(
        dataset=Sqf(split=Sqf.Splits.RACE),
        samples=12_347,
        x_features=145,
        discrete_features=139,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="SQF Race",
        sum_s=6_471,
        sum_y=1_289,
    ),
    DT(
        dataset=Sqf(split=Sqf.Splits.RACE_SEX),
        samples=12_347,
        x_features=144,
        discrete_features=138,
        s_features=1,
        num_sens=4,
        y_features=1,
        num_labels=2,
        name="SQF Race-Sex",
        sum_s=24_336,
        sum_y=1_289,
    ),
    DT(
        dataset=Toy(),
        samples=400,
        x_features=10,
        discrete_features=8,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="Toy",
        sum_s=200,
        sum_y=231,
    ),
    DT(
        dataset=AcsIncome(root=Path("~/Data"), year="2018", horizon=1, states=["AL"]),
        samples=22_268,
        x_features=45,
        discrete_features=40,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="ACS_Income_2018_1_AL_Sex",
        sum_s=11_622,
        sum_y=6_924,
    ),
    DT(
        dataset=AcsIncome(root=Path("~/Data"), year="2018", horizon=1, states=["PA"]),
        samples=68_308,
        x_features=45,
        discrete_features=40,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="ACS_Income_2018_1_PA_Sex",
        sum_s=35_480,
        sum_y=24_385,
    ),
    DT(
        dataset=AcsIncome(root=Path("~/Data"), year="2018", horizon=1, states=["AL", "PA"]),
        samples=90_576,
        x_features=45,
        discrete_features=40,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="ACS_Income_2018_1_AL_PA_Sex",
        sum_s=47_102,
        sum_y=31_309,
    ),
    DT(
        dataset=AcsIncome(root=Path("~/Data"), year="2018", horizon=1, states=["AL"], split="Race"),
        samples=22_268,
        x_features=38,
        discrete_features=33,
        s_features=1,
        num_sens=9,
        y_features=1,
        num_labels=2,
        name="ACS_Income_2018_1_AL_Race",
        sum_s=9_947,
        sum_y=6_924,
    ),
    DT(
        dataset=AcsIncome(
            root=Path("~/Data"), year="2018", horizon=1, states=["AL"], split="Sex-Race"
        ),
        samples=22_268,
        x_features=36,
        discrete_features=31,
        s_features=1,
        num_sens=17,
        y_features=1,
        num_labels=2,
        name="ACS_Income_2018_1_AL_Sex-Race",
        sum_s=31_516,
        sum_y=6_924,
    ),
    DT(
        dataset=AcsEmployment(root=Path("~/Data"), year="2018", horizon=1, states=["AL"]),
        samples=47_777,
        x_features=90,
        discrete_features=89,
        s_features=1,
        num_sens=2,
        y_features=1,
        num_labels=2,
        name="ACS_Employment_2018_1_AL_Sex",
        sum_s=22_972,
        sum_y=19_575,
    ),
]


@pytest.mark.parametrize("dt", dts, ids=idfn)
@pytest.mark.xdist_group("data_files")
def test_data_shape(dt: DT):
    """Test loading data."""
    data: DataTuple = dt.dataset.load()
    assert (dt.samples, dt.x_features) == data.x.shape
    assert (dt.samples,) == data.s.shape
    assert (dt.samples,) == data.y.shape

    assert len(dt.dataset.feature_split()["x"]) == dt.x_features
    assert len(dt.dataset.discrete_features) == dt.discrete_features
    assert len(dt.dataset.continuous_features) == (dt.x_features - dt.discrete_features)

    assert data.s.nunique() == dt.num_sens
    assert data.y.nunique() == dt.num_labels

    assert data.s.sum() == dt.sum_s
    assert data.y.sum() == dt.sum_y

    assert data.name == dt.name

    data: DataTuple = dt.dataset.load()
    assert (dt.samples, dt.x_features) == data.x.shape
    assert (dt.samples,) == data.s.shape
    assert (dt.samples,) == data.y.shape

    assert (
        len(flatten_dict(dt.dataset.disc_feature_groups)) + len(dt.dataset.continuous_features)
    ) == len(data.x.columns)


@pytest.mark.parametrize("fair", [False, True])
@pytest.mark.parametrize(
    "target", [Synthetic.Targets.Y1, Synthetic.Targets.Y2, Synthetic.Targets.Y3]
)
@pytest.mark.parametrize(
    "scenario",
    [
        Synthetic.Scenarios.S1,
        Synthetic.Scenarios.S2,
        Synthetic.Scenarios.S3,
        Synthetic.Scenarios.S4,
    ],
)
@pytest.mark.parametrize("samples", [10, 100, 1_000])
def test_synth_data_shape(
    scenario: SyntheticScenarios, target: SyntheticTargets, fair: bool, samples: int
):
    """Test loading data."""
    dataset = Synthetic(scenario=scenario, target=target, fair=fair, num_samples=samples)
    data: DataTuple = dataset.load()
    assert (samples, 4) == data.x.shape
    assert (samples,) == data.s.shape
    assert (samples,) == data.y.shape

    assert len(dataset.feature_split()["x"]) == 4
    assert len(dataset.discrete_features) == 0
    assert len(dataset.continuous_features) == 4

    assert data.s.nunique() == 2
    assert data.y.nunique() == 2

    if fair:
        assert data.name == f"Synthetic - Scenario {scenario.value}, target {target.value} fair"
    else:
        assert data.name == f"Synthetic - Scenario {scenario.value}, target {target.value}"

    data: DataTuple = dataset.load()
    assert (samples, 4) == data.x.shape
    assert (samples,) == data.s.shape
    assert (samples,) == data.y.shape


def test_load_data_as_a_function(data_root: Path):
    """Test load data as a function."""
    data_loc = data_root / "toy.csv"
    data_obj: Dataset = create_data_obj(data_loc, s_column="sensitive-attr", y_column="decision")
    assert data_obj is not None
    assert data_obj.feature_split()["x"] == [
        "a1",
        "a2",
        "disc_1_a",
        "disc_1_b",
        "disc_1_c",
        "disc_1_d",
        "disc_1_e",
        "disc_2_x",
        "disc_2_y",
        "disc_2_z",
    ]
    assert data_obj.feature_split()["s"] == ["sensitive-attr"]
    assert data_obj.feature_split()["y"] == ["decision"]
    assert len(data_obj) == 400


def test_joining_2_load_functions(data_root: Path):
    """Test joining 2 load functions."""
    data_loc = data_root / "toy.csv"
    data_obj: Dataset = create_data_obj(data_loc, s_column="sensitive-attr", y_column="decision")
    data: DataTuple = data_obj.load()
    assert (400, 10) == data.x.shape
    assert (400,) == data.s.shape
    assert (400,) == data.y.shape


def test_load_compas_feature_length():
    """Test load compas feature length."""
    data: DataTuple = Compas().load()
    assert len(Compas().feature_split()["x"]) == 400
    assert len(Compas().discrete_features) == 395
    assert len(Compas().continuous_features) == 5
    disc_feature_groups = Compas().disc_feature_groups
    assert disc_feature_groups is not None
    assert len(disc_feature_groups["c-charge-desc"]) == 389
    assert data.s.shape == (6167,)
    assert data.y.shape == (6167,)


def test_load_adult_explicitly_sex():
    """Test load adult explicitly sex."""
    adult_sex = Adult(split=Adult.Splits.SEX)
    data: DataTuple = adult_sex.load()
    assert (45222, 101) == data.x.shape
    assert (45222,) == data.s.shape
    assert (45222,) == data.y.shape
    assert adult_sex.disc_feature_groups is not None
    assert "sex" not in adult_sex.disc_feature_groups
    assert "salary" not in adult_sex.disc_feature_groups


def test_load_adult_race():
    """Test load adult race."""
    adult_race = Adult(split=Adult.Splits.RACE)
    data: DataTuple = adult_race.load()
    assert (45222, 98) == data.x.shape
    assert (45222,) == data.s.shape
    assert data.s.nunique() == 5
    assert (45222,) == data.y.shape
    assert adult_race.disc_feature_groups is not None
    assert "race" not in adult_race.disc_feature_groups
    assert "salary" not in adult_race.disc_feature_groups


def test_load_adult_race_sex():
    """Test load adult race sex."""
    adult_race_sex = Adult(split=Adult.Splits.RACE_SEX)
    data: DataTuple = adult_race_sex.load()
    assert (45222, 96) == data.x.shape
    assert (45222,) == data.s.shape
    assert data.s.nunique() == 2 * 5
    assert (45222,) == data.y.shape
    assert adult_race_sex.disc_feature_groups is not None
    assert "race" not in adult_race_sex.disc_feature_groups
    assert "sex" not in adult_race_sex.disc_feature_groups
    assert "salary" not in adult_race_sex.disc_feature_groups


def test_load_adult_drop_native():
    """Test load adult drop native."""
    adult_data = Adult(split=Adult.Splits.SEX, binarize_nationality=True)
    assert adult_data.name == "Adult Sex, binary nationality"
    assert "native-country_United-States" in adult_data.discrete_features
    assert "native-country_Canada" not in adult_data.discrete_features

    # with dummies
    data = adult_data.load()
    assert (45222, 62) == data.x.shape
    assert (45222,) == data.s.shape
    assert (45222,) == data.y.shape
    assert "native-country_United-States" in data.x.columns
    # the dummy feature *is* in the actual dataframe:
    assert "native-country_not_United-States" in data.x.columns
    assert "native-country_Canada" not in data.x.columns
    native_cols = data.x[["native-country_United-States", "native-country_not_United-States"]]
    assert (native_cols.sum(axis="columns") == 1).all()

    # with dummies, not ordered
    data = adult_data.load()
    assert (45222, 62) == data.x.shape
    assert (45222,) == data.s.shape
    assert (45222,) == data.y.shape
    assert "native-country_United-States" in data.x.columns
    # the dummy feature *is* in the actual dataframe:
    assert "native-country_not_United-States" in data.x.columns
    assert "native-country_Canada" not in data.x.columns
    native_cols = data.x[["native-country_United-States", "native-country_not_United-States"]]
    assert (native_cols.sum(axis="columns") == 1).all()


def test_load_adult_education():
    """Test load adult education."""
    adult_data = Adult(split=Adult.Splits.EDUCTAION)
    assert adult_data.name == "Adult Education"
    assert "education_HS-grad" in adult_data.sens_attrs
    assert "education_other" in adult_data.sens_attrs
    assert "education_Masters" not in adult_data.sens_attrs

    # ordered
    data = adult_data.load()
    assert (45222, 86) == data.x.shape
    assert (45222,) == data.s.shape
    assert data.s.nunique() == 3
    assert (45222,) == data.y.shape
    assert "education" == data.s.name

    # not ordered
    data = adult_data.load()
    assert (45222, 86) == data.x.shape
    assert (45222,) == data.s.shape
    assert data.s.nunique() == 3
    assert (45222,) == data.y.shape
    assert "education" == data.s.name


def test_load_adult_education_drop():
    """Test load adult education."""
    adult_data = Adult(split=Adult.Splits.EDUCTAION, binarize_nationality=True)
    assert adult_data.name == "Adult Education, binary nationality"
    assert "education_HS-grad" in adult_data.sens_attrs
    assert "education_other" in adult_data.sens_attrs
    assert "education_Masters" not in adult_data.sens_attrs

    # ordered
    data = adult_data.load()
    assert (45222, 47) == data.x.shape
    assert (45222,) == data.s.shape
    assert data.s.nunique() == 3
    assert (45222,) == data.y.shape
    assert "education" == data.s.name

    # not ordered
    data = adult_data.load()
    assert (45222, 47) == data.x.shape
    assert (45222,) == data.s.shape
    assert data.s.nunique() == 3
    assert (45222,) == data.y.shape
    assert "education" == data.s.name


def test_additional_columns_load(data_root: Path):
    """Test additional columns load."""
    data_loc = data_root / "adult.csv.zip"
    data_obj: Dataset = create_data_obj(
        data_loc,
        s_column="race_White",
        y_column="salary_>50K",
        additional_to_drop=["race_Black", "salary_<=50K"],
    )
    data: DataTuple = data_obj.load()

    assert (45222, 102) == data.x.shape
    assert (45222,) == data.s.shape
    assert (45222,) == data.y.shape


def test_domain_adapt_adult():
    """Test domain adapt adult."""
    data: DataTuple = Adult().load()
    train, test = em.domain_split(
        datatup=data,
        tr_cond="education_Masters == 0. & education_Doctorate == 0.",
        te_cond="education_Masters == 1. | education_Doctorate == 1.",
    )
    assert (39106, 101) == train.x.shape
    assert (39106,) == train.s.shape
    assert (39106,) == train.y.shape

    assert (6116, 101) == test.x.shape
    assert (6116,) == test.s.shape
    assert (6116,) == test.y.shape

    data = Adult().load()
    train, test = em.domain_split(
        datatup=data, tr_cond="education_Masters == 0.", te_cond="education_Masters == 1."
    )
    assert (40194, 101) == train.x.shape
    assert (40194,) == train.s.shape
    assert (40194,) == train.y.shape

    assert (5028, 101) == test.x.shape
    assert (5028,) == test.s.shape
    assert (5028,) == test.y.shape

    data = Adult().load()
    train, test = em.domain_split(
        datatup=data,
        tr_cond="education_Masters == 0. & education_Doctorate == 0. & education_Bachelors == 0.",
        te_cond="education_Masters == 1. | education_Doctorate == 1. | education_Bachelors == 1.",
    )
    assert (23966, 101) == train.x.shape
    assert (23966,) == train.s.shape
    assert (23966,) == train.y.shape

    assert (21256, 101) == test.x.shape
    assert (21256,) == test.s.shape
    assert (21256,) == test.y.shape


def test_query():
    """Test query."""
    x: pd.DataFrame = pd.DataFrame(columns=["0a", "b"], data=[[0, 1], [2, 3], [4, 5]])
    s: pd.Series = pd.Series(name="c=", data=[6, 7, 8])
    y: pd.Series = pd.Series(name="d", data=[9, 10, 11])
    data = DataTuple.from_df(x=x, s=s, y=y, name="test_data")
    selected = em.query_dt(data, "`0a` == 0 & `c=` == 6 & d == 9")
    pd.testing.assert_frame_equal(selected.x, x.head(1))
    pd.testing.assert_series_equal(selected.s, s.head(1))
    pd.testing.assert_series_equal(selected.y, y.head(1))


def test_concat():
    """Test concat."""
    x: pd.DataFrame = pd.DataFrame(columns=["a"], data=[[1]])
    s: pd.Series = pd.Series(name="b", data=[2])
    y: pd.Series = pd.Series(name="c", data=[3])
    data1 = DataTuple.from_df(x=x, s=s, y=y, name="test_data")
    x = pd.DataFrame(columns=["a"], data=[4])
    s = pd.Series(name="b", data=[5])
    y = pd.Series(name="c", data=[6])
    data2 = DataTuple.from_df(x=x, s=s, y=y, name="test_tuple")
    data3 = em.concat([data1, data2], ignore_index=True)
    pd.testing.assert_frame_equal(data3.x, pd.DataFrame(columns=["a"], data=[[1], [4]]))
    pd.testing.assert_series_equal(data3.s, pd.Series(name="b", data=[2, 5]))
    pd.testing.assert_series_equal(data3.y, pd.Series(name="c", data=[3, 6]))


def test_group_prefixes():
    """Test group prefixes."""
    names = ["a_asf", "a_fds", "good_lhdf", "good_dsdw", "sas"]
    grouped_indices = group_disc_feat_indices(names, prefix_sep="_")

    assert len(grouped_indices) == 3
    assert grouped_indices[0] == slice(0, 2)
    assert grouped_indices[1] == slice(2, 4)
    assert grouped_indices[2] == slice(4, 5)


def test_expand_s():
    """Test expanding s."""
    sens_attr_spec = {
        "Gender": LabelGroup(["Female", "Male"], multiplier=3),
        "Race": LabelGroup(["Blue", "Green", "Pink"], multiplier=1),
    }
    data = LegacyDataset(
        name="test",
        filename_or_path="non-existent",
        features=[],
        cont_features=[],
        sens_attr_spec=sens_attr_spec,
        class_label_spec="label",
        num_samples=7,
    )

    compact_df = pd.Series([0, 4, 3, 1, 3, 5, 2], name="Gender,Race")
    gender_expanded = pd.DataFrame(
        [[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0]], columns=["Female", "Male"]
    )
    race_expanded = pd.DataFrame(
        [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],
        columns=["Blue", "Green", "Pink"],
    )
    levels: Mapping[str, pd.DataFrame] = {"Race": race_expanded, "Gender": gender_expanded}
    multilevel_df = pd.concat(levels, axis="columns")
    raw_df = pd.concat([gender_expanded, race_expanded], axis="columns")

    pd.testing.assert_series_equal(
        data._one_hot_encode_and_combine(raw_df, sens_attr_spec)[0], compact_df
    )
    pd.testing.assert_frame_equal(
        data.expand_labels(compact_df, "s").astype("int64"), multilevel_df
    )


def test_simple_spec():
    """Test the simple spec function."""
    sens_attrs = {"race": ["blue", "green", "pink"], "gender": ["female", "male"]}
    spec = spec_from_binary_cols(sens_attrs)
    assert spec == {
        "gender": LabelGroup(["female", "male"], multiplier=3),
        "race": LabelGroup(["blue", "green", "pink"], multiplier=1),
    }


@pytest.mark.slow
@pytest.mark.parametrize("data", [Adult, Admissions, Compas, Credit, Crime, German])
def test_aif_conversion(data: Callable[[], LegacyDataset]):
    """Load a dataset in AIF form.

    There might be a case where you want to load an EthicML dataset
    and use it with a model that exists in AIF360 that we don't have in EthicML.
    """
    data().load_aif()
