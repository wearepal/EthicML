"""Test the loading data capability."""
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest

import ethicml as em
from ethicml import Admissions, Compas, Credit, Crime, DataTuple, German, LoadableDataset
from ethicml.data.tabular_data.adult import Adult, AdultSplits
from ethicml.data.util import flatten_dict


def test_can_load_test_data(data_root: Path):
    """Test whether we can load test data."""
    data_loc = data_root / "toy.csv"
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


class DT(NamedTuple):
    """Describe data for the tests."""

    dataset: em.Dataset
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


@pytest.mark.parametrize(
    "dt",
    [
        DT(
            dataset=em.admissions(),
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
            dataset=em.admissions(split="Gender", invert_s=True),
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
            dataset=em.adult(),
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
            dataset=em.adult("Sex", binarize_nationality=True),
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
            dataset=em.adult("Sex", binarize_race=True),
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
            dataset=em.adult("Sex", binarize_nationality=True, binarize_race=True),
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
            dataset=em.adult(split="Sex"),
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
            dataset=em.adult(split=AdultSplits.SEX),
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
            dataset=em.adult(split="Race"),
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
            dataset=em.adult(split=AdultSplits.RACE),
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
            dataset=em.adult(split="Race-Binary"),
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
            dataset=em.adult(split="Nationality"),
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
            dataset=em.adult(split="Education"),
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
            dataset=em.compas(),
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
            dataset=em.compas(split="Sex"),
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
            dataset=em.compas(split="Race"),
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
            dataset=em.compas(split="Race-Sex"),
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
            dataset=em.credit(),
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
            dataset=em.credit(split="Sex"),
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
            dataset=em.crime(),
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
            dataset=em.crime(split="Race-Binary"),
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
            dataset=em.german(),
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
            dataset=em.german(split="Sex"),
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
            dataset=em.health(),
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
            dataset=em.health(split="Sex"),
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
            dataset=em.law(split="Sex"),
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
            dataset=em.law(split="Race"),
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
            dataset=em.law(split="Sex-Race"),
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
            dataset=em.lipton(),
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
            dataset=em.nonbinary_toy(),
            samples=100,
            x_features=2,
            discrete_features=0,
            s_features=1,
            num_sens=2,
            y_features=1,
            num_labels=5,
            name="NonBinaryToy",
            sum_s=48,
            sum_y=300,
        ),
        DT(
            dataset=em.sqf(),
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
            dataset=em.sqf(split="Sex"),
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
            dataset=em.sqf(split="Race"),
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
            dataset=em.sqf(split="Race-Sex"),
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
            dataset=em.toy(),
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
            dataset=em.acs_income(root=Path("~/Data"), year="2018", horizon=1, states=["AL"]),
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
            dataset=em.acs_income(root=Path("~/Data"), year="2018", horizon=1, states=["PA"]),
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
            dataset=em.acs_income(root=Path("~/Data"), year="2018", horizon=1, states=["AL", "PA"]),
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
            dataset=em.acs_income(
                root=Path("~/Data"), year="2018", horizon=1, states=["AL"], split="Race"
            ),
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
            dataset=em.acs_income(
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
            dataset=em.acs_employment(root=Path("~/Data"), year="2018", horizon=1, states=["AL"]),
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
    ],
    ids=idfn,
)
def test_data_shape(dt: DT):
    """Test loading data."""
    data: DataTuple = dt.dataset.load()
    assert (dt.samples, dt.x_features) == data.x.shape
    assert (dt.samples, dt.s_features) == data.s.shape
    assert (dt.samples, dt.y_features) == data.y.shape

    assert len(dt.dataset.ordered_features["x"]) == dt.x_features
    assert len(dt.dataset.discrete_features) == dt.discrete_features
    assert len(dt.dataset.continuous_features) == (dt.x_features - dt.discrete_features)

    assert data.s.nunique()[0] == dt.num_sens  # type: ignore[comparison-overlap]
    assert data.y.nunique()[0] == dt.num_labels  # type: ignore[comparison-overlap]

    assert data.s.sum().values[0] == dt.sum_s
    assert data.y.sum().values[0] == dt.sum_y

    assert data.name == dt.name

    data: DataTuple = dt.dataset.load(ordered=True)
    assert (dt.samples, dt.x_features) == data.x.shape
    assert (dt.samples, dt.s_features) == data.s.shape
    assert (dt.samples, dt.y_features) == data.y.shape

    assert (
        len(flatten_dict(dt.dataset.disc_feature_groups)) + len(dt.dataset.continuous_features)
    ) == len(data.x.columns)


@pytest.mark.parametrize("fair", [False, True])
@pytest.mark.parametrize("target", [1, 2, 3])
@pytest.mark.parametrize("scenario", [1, 2, 3, 4])
@pytest.mark.parametrize("samples", [10, 100, 1_000])
def test_synth_data_shape(scenario, target, fair, samples):
    """Test loading data."""
    dataset = em.synthetic(scenario=scenario, target=target, fair=fair, num_samples=samples)
    data: DataTuple = dataset.load()
    assert (samples, 4) == data.x.shape
    assert (samples, 1) == data.s.shape
    assert (samples, 1) == data.y.shape

    assert len(dataset.ordered_features["x"]) == 4
    assert len(dataset.discrete_features) == 0
    assert len(dataset.continuous_features) == 4

    assert data.s.nunique()[0] == 2  # type: ignore[comparison-overlap]
    assert data.y.nunique()[0] == 2  # type: ignore[comparison-overlap]

    if fair:
        assert data.name == f"Synthetic - Scenario {scenario}, target {target} fair"
    else:
        assert data.name == f"Synthetic - Scenario {scenario}, target {target}"

    data: DataTuple = dataset.load(ordered=True)
    assert (samples, 4) == data.x.shape
    assert (samples, 1) == data.s.shape
    assert (samples, 1) == data.y.shape


def test_load_data_as_a_function(data_root: Path):
    """Test load data as a function."""
    data_loc = data_root / "toy.csv"
    data_obj: em.Dataset = em.create_data_obj(
        data_loc, s_column="sensitive-attr", y_column="decision"
    )
    assert data_obj is not None
    assert data_obj.feature_split["x"] == [
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
    assert data_obj.feature_split["s"] == ["sensitive-attr"]
    assert data_obj.feature_split["y"] == ["decision"]
    assert len(data_obj) == 400


def test_joining_2_load_functions(data_root: Path):
    """Test joining 2 load functions."""
    data_loc = data_root / "toy.csv"
    data_obj: em.Dataset = em.create_data_obj(
        data_loc, s_column="sensitive-attr", y_column="decision"
    )
    data: DataTuple = data_obj.load()
    assert (400, 10) == data.x.shape
    assert (400, 1) == data.s.shape
    assert (400, 1) == data.y.shape


def test_load_compas_feature_length():
    """Test load compas feature length."""
    data: DataTuple = em.compas().load()
    assert len(em.compas().ordered_features["x"]) == 400
    assert len(em.compas().discrete_features) == 395
    assert len(em.compas().continuous_features) == 5
    disc_feature_groups = em.compas().disc_feature_groups
    assert disc_feature_groups is not None
    assert len(disc_feature_groups["c-charge-desc"]) == 389
    assert data.s.shape == (6167, 1)
    assert data.y.shape == (6167, 1)


def test_load_adult_explicitly_sex():
    """Test load adult explicitly sex."""
    adult_sex = em.adult("Sex")
    data: DataTuple = adult_sex.load()
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert adult_sex.disc_feature_groups is not None
    assert "sex" not in adult_sex.disc_feature_groups
    assert "salary" not in adult_sex.disc_feature_groups


def test_load_adult_race():
    """Test load adult race."""
    adult_race = em.adult("Race")
    data: DataTuple = adult_race.load()
    assert (45222, 98) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert data.s.nunique()[0] == 5  # type: ignore[comparison-overlap]
    assert (45222, 1) == data.y.shape
    assert adult_race.disc_feature_groups is not None
    assert "race" not in adult_race.disc_feature_groups
    assert "salary" not in adult_race.disc_feature_groups


def test_load_adult_race_sex():
    """Test load adult race sex."""
    adult_race_sex = em.adult("Race-Sex")
    data: DataTuple = adult_race_sex.load()
    assert (45222, 96) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert data.s.nunique()[0] == 2 * 5  # type: ignore[comparison-overlap]
    assert (45222, 1) == data.y.shape
    assert adult_race_sex.disc_feature_groups is not None
    assert "race" not in adult_race_sex.disc_feature_groups
    assert "sex" not in adult_race_sex.disc_feature_groups
    assert "salary" not in adult_race_sex.disc_feature_groups


def test_race_feature_split():
    """Test race feature split."""
    adult_data: em.Dataset = em.adult(split="Custom")
    adult_data._sens_attr_spec = "race_White"
    adult_data._s_prefix = ["race"]
    adult_data._class_label_spec = "salary_>50K"
    adult_data._class_label_prefix = ["salary"]

    data: DataTuple = adult_data.load()

    assert (45222, 98) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_load_adult_drop_native():
    """Test load adult drop native."""
    adult_data = em.adult("Sex", binarize_nationality=True)
    assert adult_data.name == "Adult Sex, binary nationality"
    assert "native-country_United-States" in adult_data.discrete_features
    assert "native-country_Canada" not in adult_data.discrete_features

    # with dummies
    data = adult_data.load(ordered=True)
    assert (45222, 62) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert "native-country_United-States" in data.x.columns
    # the dummy feature *is* in the actual dataframe:
    assert "native-country_not_United-States" in data.x.columns
    assert "native-country_Canada" not in data.x.columns
    native_cols = data.x[["native-country_United-States", "native-country_not_United-States"]]
    assert (native_cols.sum(axis="columns") == 1).all()

    # with dummies, not ordered
    data = adult_data.load(ordered=False)
    assert (45222, 62) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert "native-country_United-States" in data.x.columns
    # the dummy feature *is* in the actual dataframe:
    assert "native-country_not_United-States" in data.x.columns
    assert "native-country_Canada" not in data.x.columns
    native_cols = data.x[["native-country_United-States", "native-country_not_United-States"]]
    assert (native_cols.sum(axis="columns") == 1).all()


def test_load_adult_education():
    """Test load adult education."""
    adult_data = em.adult("Education")
    assert adult_data.name == "Adult Education"
    assert "education_HS-grad" in adult_data.sens_attrs
    assert "education_other" in adult_data.sens_attrs
    assert "education_Masters" not in adult_data.sens_attrs

    # ordered
    data = adult_data.load(ordered=True)
    assert (45222, 86) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert data.s.nunique()[0] == 3
    assert (45222, 1) == data.y.shape
    assert "education" in data.s.columns

    # not ordered
    data = adult_data.load()
    assert (45222, 86) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert data.s.nunique()[0] == 3
    assert (45222, 1) == data.y.shape
    assert "education" in data.s.columns


def test_load_adult_education_drop():
    """Test load adult education."""
    adult_data = em.adult("Education", binarize_nationality=True)
    assert adult_data.name == "Adult Education, binary nationality"
    assert "education_HS-grad" in adult_data.sens_attrs
    assert "education_other" in adult_data.sens_attrs
    assert "education_Masters" not in adult_data.sens_attrs

    # ordered
    data = adult_data.load(ordered=True)
    assert (45222, 47) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert data.s.nunique()[0] == 3
    assert (45222, 1) == data.y.shape
    assert "education" in data.s.columns

    # not ordered
    data = adult_data.load()
    assert (45222, 47) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert data.s.nunique()[0] == 3
    assert (45222, 1) == data.y.shape
    assert "education" in data.s.columns


def test_additional_columns_load(data_root: Path):
    """Test additional columns load."""
    data_loc = data_root / "adult.csv.zip"
    data_obj: em.Dataset = em.create_data_obj(
        data_loc,
        s_column="race_White",
        y_column="salary_>50K",
        additional_to_drop=["race_Black", "salary_<=50K"],
    )
    data: DataTuple = data_obj.load()

    assert (45222, 102) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_domain_adapt_adult():
    """Test domain adapt adult."""
    data: DataTuple = em.adult().load()
    train, test = em.domain_split(
        datatup=data,
        tr_cond="education_Masters == 0. & education_Doctorate == 0.",
        te_cond="education_Masters == 1. | education_Doctorate == 1.",
    )
    assert (39106, 101) == train.x.shape
    assert (39106, 1) == train.s.shape
    assert (39106, 1) == train.y.shape

    assert (6116, 101) == test.x.shape
    assert (6116, 1) == test.s.shape
    assert (6116, 1) == test.y.shape

    data = em.adult().load()
    train, test = em.domain_split(
        datatup=data, tr_cond="education_Masters == 0.", te_cond="education_Masters == 1."
    )
    assert (40194, 101) == train.x.shape
    assert (40194, 1) == train.s.shape
    assert (40194, 1) == train.y.shape

    assert (5028, 101) == test.x.shape
    assert (5028, 1) == test.s.shape
    assert (5028, 1) == test.y.shape

    data = em.adult().load()
    train, test = em.domain_split(
        datatup=data,
        tr_cond="education_Masters == 0. & education_Doctorate == 0. & education_Bachelors == 0.",
        te_cond="education_Masters == 1. | education_Doctorate == 1. | education_Bachelors == 1.",
    )
    assert (23966, 101) == train.x.shape
    assert (23966, 1) == train.s.shape
    assert (23966, 1) == train.y.shape

    assert (21256, 101) == test.x.shape
    assert (21256, 1) == test.s.shape
    assert (21256, 1) == test.y.shape


def test_query():
    """Test query."""
    x: pd.DataFrame = pd.DataFrame(columns=["0a", "b"], data=[[0, 1], [2, 3], [4, 5]])
    s: pd.DataFrame = pd.DataFrame(columns=["c="], data=[[6], [7], [8]])
    y: pd.DataFrame = pd.DataFrame(columns=["d"], data=[[9], [10], [11]])
    data = DataTuple(x=x, s=s, y=y, name="test_data")
    selected = em.query_dt(data, "`0a` == 0 & `c=` == 6 & d == 9")
    pd.testing.assert_frame_equal(selected.x, x.head(1))
    pd.testing.assert_frame_equal(selected.s, s.head(1))
    pd.testing.assert_frame_equal(selected.y, y.head(1))


def test_concat():
    """Test concat."""
    x: pd.DataFrame = pd.DataFrame(columns=["a"], data=[[1]])
    s: pd.DataFrame = pd.DataFrame(columns=["b"], data=[[2]])
    y: pd.DataFrame = pd.DataFrame(columns=["c"], data=[[3]])
    data1 = DataTuple(x=x, s=s, y=y, name="test_data")
    x = pd.DataFrame(columns=["a"], data=[[4]])
    s = pd.DataFrame(columns=["b"], data=[[5]])
    y = pd.DataFrame(columns=["c"], data=[[6]])
    data2 = DataTuple(x=x, s=s, y=y, name="test_tuple")
    data3 = em.concat_dt([data1, data2], axis="index", ignore_index=True)
    pd.testing.assert_frame_equal(data3.x, pd.DataFrame(columns=["a"], data=[[1], [4]]))
    pd.testing.assert_frame_equal(data3.s, pd.DataFrame(columns=["b"], data=[[2], [5]]))
    pd.testing.assert_frame_equal(data3.y, pd.DataFrame(columns=["c"], data=[[3], [6]]))


def test_group_prefixes():
    """Test group prefixes."""
    names = ["a_asf", "a_fds", "good_lhdf", "good_dsdw", "sas"]
    grouped_indexes = em.group_disc_feat_indexes(names, prefix_sep="_")

    assert len(grouped_indexes) == 3
    assert grouped_indexes[0] == slice(0, 2)
    assert grouped_indexes[1] == slice(2, 4)
    assert grouped_indexes[2] == slice(4, 5)


def test_expand_s():
    """Test expanding s."""
    data = em.LoadableDataset(
        name="test",
        filename_or_path="non-existent",
        features=[],
        cont_features=[],
        sens_attr_spec={
            "Gender": em.LabelGroup(["Female", "Male"], multiplier=3),
            "Race": em.LabelGroup(["Blue", "Green", "Pink"], multiplier=1),
        },
        class_label_spec="label",
        num_samples=7,
        discrete_only=False,
    )

    compact_df = pd.DataFrame([0, 4, 3, 1, 3, 5, 2], columns=["Gender,Race"])
    gender_expanded = pd.DataFrame(
        [[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0]], columns=["Female", "Male"]
    )
    race_expanded = pd.DataFrame(
        [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],
        columns=["Blue", "Green", "Pink"],
    )
    multilevel_df = pd.concat({"Race": race_expanded, "Gender": gender_expanded}, axis="columns")
    raw_df = pd.concat([gender_expanded, race_expanded], axis="columns")

    pd.testing.assert_frame_equal(data._maybe_combine_labels(raw_df, "s")[0], compact_df)
    pd.testing.assert_frame_equal(
        data.expand_labels(compact_df, "s").astype("int64"), multilevel_df
    )


def test_simple_spec():
    """Test the simple spec function."""
    sens_attrs = {"race": ["blue", "green", "pink"], "gender": ["female", "male"]}
    spec = em.simple_spec(sens_attrs)
    assert spec == {
        "gender": em.LabelGroup(["female", "male"], multiplier=3),
        "race": em.LabelGroup(["blue", "green", "pink"], multiplier=1),
    }


@pytest.mark.parametrize("data", [Adult(), Admissions(), Compas(), Credit(), Crime(), German()])
def test_aif_conversion(data: LoadableDataset):
    """Load a dataset in AIF form.

    There might be a case where you want to load an EthicML dataset
    and use it with a model that exists in AIF360 that we don't have in EthicML.
    """
    data.load_aif()
