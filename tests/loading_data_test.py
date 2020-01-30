"""
Test the loading data capability
"""
from pathlib import Path
import pandas as pd

from ethicml.common import ROOT_PATH
from ethicml.data import (
    Dataset,
    load_data,
    create_data_obj,
    Adult,
    Compas,
    Credit,
    German,
    Sqf,
    Toy,
    NonBinaryToy,
    group_disc_feat_indexes,
)
from ethicml.data.crime import Crime
from ethicml.data.health import Health
from ethicml.utility import DataTuple, concat_dt
from ethicml.preprocessing import domain_split, query_dt


def test_can_load_test_data():
    """test whether we can load test data"""
    data_loc: Path = ROOT_PATH / "data" / "csvs" / "toy.csv"
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


def test_load_data():
    """test loading data"""
    data: DataTuple = load_data(Toy())
    assert (2000, 2) == data.x.shape
    assert (2000, 1) == data.s.shape
    assert (2000, 1) == data.y.shape
    assert data.name == "Toy"


def test_load_non_binary_data():
    """test load non binary data"""
    data: DataTuple = load_data(NonBinaryToy())
    assert (100, 2) == data.x.shape
    assert (100, 1) == data.s.shape
    assert (100, 1) == data.y.shape


def test_discrete_data():
    """test discrete data"""
    data: DataTuple = load_data(Adult())
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert len(Adult().discrete_features) == 96
    assert len(Adult(split="Race").discrete_features) == 93
    assert len(Adult(split="Nationality").discrete_features) == 57


def test_load_data_as_a_function():
    """test load data as a function"""
    data_loc: Path = ROOT_PATH / "data" / "csvs" / "toy.csv"
    data_obj: Dataset = create_data_obj(data_loc, s_columns=["s"], y_columns=["y"])
    assert data_obj is not None
    assert data_obj.feature_split["x"] == ["a1", "a2"]
    assert data_obj.feature_split["s"] == ["s"]
    assert data_obj.feature_split["y"] == ["y"]
    assert data_obj.filename == "toy.csv"


def test_joining_2_load_functions():
    """test joining 2 load functions"""
    data_loc: Path = ROOT_PATH / "data" / "csvs" / "toy.csv"
    data_obj: Dataset = create_data_obj(data_loc, s_columns=["s"], y_columns=["y"])
    data: DataTuple = load_data(data_obj)
    assert (2000, 2) == data.x.shape
    assert (2000, 1) == data.s.shape
    assert (2000, 1) == data.y.shape


def test_load_adult():
    """test load adult"""
    data: DataTuple = load_data(Adult())
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert data.name == "Adult Sex"


def test_load_compas():
    """test load compas"""
    data: DataTuple = load_data(Compas())
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape
    assert data.name == "Compas Sex"


def test_load_health():
    """test load health dataset"""
    data: DataTuple = load_data(Health())
    assert (171067, 130) == data.x.shape
    assert (171067, 1) == data.s.shape
    assert (171067, 1) == data.y.shape
    assert data.name == "Health"


def test_load_crime():
    """test load crime dataset"""
    data: DataTuple = load_data(Crime())
    assert (1993, 136) == data.x.shape
    assert (1993, 1) == data.s.shape
    assert (1993, 1) == data.y.shape
    assert data.name == "Crime Race-Binary"


def test_load_sqf():
    """test load sqf"""
    data: DataTuple = load_data(Sqf())
    assert (12347, 144) == data.x.shape
    assert (12347, 1) == data.s.shape
    assert (12347, 1) == data.y.shape
    assert data.name == "SQF Sex"


def test_load_german():
    """test load german"""
    data: DataTuple = load_data(German())
    assert (1000, 57) == data.x.shape
    assert (1000, 1) == data.s.shape
    assert (1000, 1) == data.y.shape
    assert data.name == "German Sex"


def test_load_german_ordered():
    """test load german ordered"""
    data: DataTuple = load_data(German(), ordered=True)
    assert (1000, 57) == data.x.shape
    assert (1000, 1) == data.s.shape
    assert (1000, 1) == data.y.shape


def test_load_compas_explicitly_sex():
    """test load compas explicitly sex"""
    data: DataTuple = load_data(Compas("Sex"))
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_compas_feature_length():
    """test load compas feature length"""
    data: DataTuple = load_data(Compas())
    assert len(Compas().ordered_features["x"]) == 400
    assert len(Compas().discrete_features) == 395
    assert len(Compas().continuous_features) == 5
    assert data.s.shape == (6167, 1)
    assert data.y.shape == (6167, 1)


def test_load_health_feature_length():
    """test load health feature length"""
    data: DataTuple = load_data(Health())
    assert len(Health().ordered_features["x"]) == 130
    assert len(Health().discrete_features) == 12
    assert len(Health().continuous_features) == 118
    assert data.s.shape == (171067, 1)
    assert data.y.shape == (171067, 1)


def test_load_crime_feature_length():
    """test load crime feature length"""
    data: DataTuple = load_data(Crime())
    assert len(Crime().ordered_features["x"]) == 136
    assert len(Crime().discrete_features) == 46
    assert len(Crime().continuous_features) == 90
    assert data.s.shape == (1993, 1)
    assert data.y.shape == (1993, 1)


def test_load_credit_feature_length():
    """test load credit feature length"""
    data: DataTuple = load_data(Credit())
    assert len(Credit().ordered_features["x"]) == 26
    assert len(Credit().discrete_features) == 6
    assert len(Credit().continuous_features) == 20
    assert data.s.shape == (30000, 1)
    assert data.y.shape == (30000, 1)


def test_load_adult_explicitly_sex():
    """test load adult explicitly sex"""
    data: DataTuple = load_data(Adult("Sex"))
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_load_adult_race():
    """test load adult race"""
    data: DataTuple = load_data(Adult("Race"))
    assert (45222, 98) == data.x.shape
    assert (45222, 5) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_load_adult_race_binary():
    """test load adult race binary"""
    data: DataTuple = load_data(Adult("Race-Binary"))
    assert (45222, 98) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert data.name == "Adult Race-Binary"


def test_load_compas_race():
    """test load compas race"""
    data: DataTuple = load_data(Compas("Race"))
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_adult_race_sex():
    """test load adult race sex"""
    data: DataTuple = load_data(Adult("Race-Sex"))
    assert (45222, 96) == data.x.shape
    assert (45222, 6) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_load_compas_race_sex():
    """test load compas race sex"""
    data: DataTuple = load_data(Compas("Race-Sex"))
    assert (6167, 399) == data.x.shape
    assert (6167, 2) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_adult_nationality():
    """test load adult nationality"""
    data: DataTuple = load_data(Adult("Nationality"))
    assert (45222, 62) == data.x.shape
    assert (45222, 41) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_race_feature_split():
    """test race feature split"""
    adult: Adult = Adult(split="Custom")
    adult.sens_attrs = ["race_White"]
    adult.s_prefix = ["race"]
    adult.class_labels = ["salary_>50K"]
    adult.class_label_prefix = ["salary"]

    data: DataTuple = load_data(adult)

    assert (45222, 98) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_load_adult_drop_native():
    """test load adult drop native"""
    adult = Adult("Sex", binarize_nationality=True)
    assert adult.name == "Adult Sex, binary nationality"
    assert "native-country_United-States" in adult.discrete_features
    assert "native-country_Canada" not in adult.discrete_features

    data: DataTuple = load_data(adult)
    assert (45222, 61) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert "native-country_United-States" in data.x.columns
    assert "native-country_Canada" not in data.x.columns


def test_additional_columns_load():
    """test additional columns load"""
    data_loc: Path = ROOT_PATH / "data" / "csvs" / "adult.csv"
    data_obj: Dataset = create_data_obj(
        data_loc,
        s_columns=["race_White"],
        y_columns=["salary_>50K"],
        additional_to_drop=["race_Black", "salary_<=50K"],
    )
    data: DataTuple = load_data(data_obj)

    assert (45222, 102) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_domain_adapt_adult():
    """test domain adapt adult"""
    data: DataTuple = load_data(Adult())
    train, test = domain_split(
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

    data = load_data(Adult())
    train, test = domain_split(
        datatup=data, tr_cond="education_Masters == 0.", te_cond="education_Masters == 1."
    )
    assert (40194, 101) == train.x.shape
    assert (40194, 1) == train.s.shape
    assert (40194, 1) == train.y.shape

    assert (5028, 101) == test.x.shape
    assert (5028, 1) == test.s.shape
    assert (5028, 1) == test.y.shape

    data = load_data(Adult())
    train, test = domain_split(
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
    """test query"""
    x: pd.DataFrame = pd.DataFrame(columns=["0a", "b"], data=[[0, 1], [2, 3], [4, 5]])
    s: pd.DataFrame = pd.DataFrame(columns=["c="], data=[[6], [7], [8]])
    y: pd.DataFrame = pd.DataFrame(columns=["d"], data=[[9], [10], [11]])
    data = DataTuple(x=x, s=s, y=y, name="test_data")
    selected = query_dt(data, "_0a == 0 & c_eq_ == 6 & d == 9")
    pd.testing.assert_frame_equal(selected.x, x.head(1))
    pd.testing.assert_frame_equal(selected.s, s.head(1))
    pd.testing.assert_frame_equal(selected.y, y.head(1))


def test_concat():
    """test concat"""
    x: pd.DataFrame = pd.DataFrame(columns=["a"], data=[[1]])
    s: pd.DataFrame = pd.DataFrame(columns=["b"], data=[[2]])
    y: pd.DataFrame = pd.DataFrame(columns=["c"], data=[[3]])
    data1 = DataTuple(x=x, s=s, y=y, name="test_data")
    x = pd.DataFrame(columns=["a"], data=[[4]])
    s = pd.DataFrame(columns=["b"], data=[[5]])
    y = pd.DataFrame(columns=["c"], data=[[6]])
    data2 = DataTuple(x=x, s=s, y=y, name="test_tuple")
    data3 = concat_dt([data1, data2], axis="index", ignore_index=True)
    pd.testing.assert_frame_equal(data3.x, pd.DataFrame(columns=["a"], data=[[1], [4]]))
    pd.testing.assert_frame_equal(data3.s, pd.DataFrame(columns=["b"], data=[[2], [5]]))
    pd.testing.assert_frame_equal(data3.y, pd.DataFrame(columns=["c"], data=[[3], [6]]))


def test_group_prefixes():
    """test group prefixes"""
    names = ["a_asf", "a_fds", "good_lhdf", "good_dsdw", "sas"]
    grouped_indexes = group_disc_feat_indexes(names, prefix_sep="_")

    assert len(grouped_indexes) == 3
    assert grouped_indexes[0] == slice(0, 2)
    assert grouped_indexes[1] == slice(2, 4)
    assert grouped_indexes[2] == slice(4, 5)
