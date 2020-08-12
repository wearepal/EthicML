"""Test the loading data capability."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ethicml.data import (
    Adult,
    Compas,
    Dataset,
    Toy,
    adult,
    celeba,
    compas,
    create_data_obj,
    credit,
    crime,
    genfaces,
    german,
    group_disc_feat_indexes,
    health,
    nonbinary_toy,
    sqf,
)
from ethicml.preprocessing import domain_split, query_dt
from ethicml.utility import DataTuple, concat_dt


def test_can_load_test_data(data_root: Path):
    """test whether we can load test data"""
    data_loc = data_root / "toy.csv"
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


def test_load_data():
    """test loading data"""
    with pytest.deprecated_call():  # assert that this gives a deprecation warning
        data: DataTuple = Toy().load()
    assert (400, 10) == data.x.shape
    assert (400, 1) == data.s.shape
    assert (400, 1) == data.y.shape
    assert data.name == "Toy"


def test_load_non_binary_data():
    """test load non binary data"""
    data: DataTuple = nonbinary_toy().load()
    assert (100, 2) == data.x.shape
    assert (100, 1) == data.s.shape
    assert (100, 1) == data.y.shape


def test_discrete_data():
    """test discrete data"""
    with pytest.deprecated_call():  # assert that this gives a deprecation warning
        data: DataTuple = Adult().load()
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert len(adult().discrete_features) == 96
    assert len(adult(split="Race").discrete_features) == 93
    assert len(adult(split="Nationality").discrete_features) == 57


def test_load_data_as_a_function(data_root: Path):
    """test load data as a function"""
    data_loc = data_root / "toy.csv"
    data_obj: Dataset = create_data_obj(
        data_loc, s_columns=["sensitive-attr"], y_columns=["decision"]
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
    """test joining 2 load functions"""
    data_loc = data_root / "toy.csv"
    data_obj: Dataset = create_data_obj(
        data_loc, s_columns=["sensitive-attr"], y_columns=["decision"]
    )
    data: DataTuple = data_obj.load()
    assert (400, 10) == data.x.shape
    assert (400, 1) == data.s.shape
    assert (400, 1) == data.y.shape


def test_load_adult():
    """test load adult"""
    data: DataTuple = adult().load()
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert data.name == "Adult Sex"


def test_load_compas():
    """test load compas"""
    with pytest.deprecated_call():  # assert that this gives a deprecation warning
        data: DataTuple = Compas().load()
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape
    assert data.name == "Compas Sex"


def test_load_health():
    """test load health dataset"""
    data: DataTuple = health().load()
    assert (171067, 130) == data.x.shape
    assert (171067, 1) == data.s.shape
    assert (171067, 1) == data.y.shape
    assert data.name == "Health"


def test_load_crime():
    """test load crime dataset"""
    data: DataTuple = crime().load()
    assert (1993, 136) == data.x.shape
    assert (1993, 1) == data.s.shape
    assert (1993, 1) == data.y.shape
    assert data.name == "Crime Race-Binary"


def test_load_sqf():
    """test load sqf"""
    data: DataTuple = sqf().load()
    assert (12347, 144) == data.x.shape
    assert (12347, 1) == data.s.shape
    assert (12347, 1) == data.y.shape
    assert data.name == "SQF Sex"


def test_load_german():
    """test load german"""
    data: DataTuple = german().load()
    assert (1000, 57) == data.x.shape
    assert (1000, 1) == data.s.shape
    assert (1000, 1) == data.y.shape
    assert data.name == "German Sex"


def test_load_german_ordered():
    """test load german ordered"""
    data: DataTuple = german().load(ordered=True)
    assert (1000, 57) == data.x.shape
    assert (1000, 1) == data.s.shape
    assert (1000, 1) == data.y.shape


def test_load_compas_explicitly_sex():
    """test load compas explicitly sex"""
    data: DataTuple = compas("Sex").load()
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_compas_feature_length():
    """test load compas feature length"""
    data: DataTuple = compas().load()
    assert len(compas().ordered_features["x"]) == 400
    assert len(compas().discrete_features) == 395
    assert len(compas().continuous_features) == 5
    disc_feature_groups = compas().disc_feature_groups
    assert disc_feature_groups is not None
    assert len(disc_feature_groups["c-charge-desc"]) == 389
    assert data.s.shape == (6167, 1)
    assert data.y.shape == (6167, 1)


def test_load_health_feature_length():
    """test load health feature length"""
    data: DataTuple = health().load()
    assert len(health().ordered_features["x"]) == 130
    assert len(health().discrete_features) == 12
    assert len(health().continuous_features) == 118
    assert data.s.shape == (171067, 1)
    assert data.y.shape == (171067, 1)


def test_load_crime_feature_length():
    """test load crime feature length"""
    data: DataTuple = crime().load()
    assert len(crime().ordered_features["x"]) == 136
    assert len(crime().discrete_features) == 46
    assert len(crime().continuous_features) == 90
    assert data.s.shape == (1993, 1)
    assert data.y.shape == (1993, 1)


def test_load_credit_feature_length():
    """test load credit feature length"""
    data: DataTuple = credit().load()
    assert len(credit().ordered_features["x"]) == 26
    assert len(credit().discrete_features) == 6
    assert len(credit().continuous_features) == 20
    assert data.s.shape == (30000, 1)
    assert data.y.shape == (30000, 1)


def test_load_adult_explicitly_sex():
    """test load adult explicitly sex"""
    adult_sex = adult("Sex")
    data: DataTuple = adult_sex.load()
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert adult_sex.disc_feature_groups is not None
    assert "sex" not in adult_sex.disc_feature_groups
    assert "salary" not in adult_sex.disc_feature_groups


def test_load_adult_race():
    """test load adult race"""
    adult_race = adult("Race")
    data: DataTuple = adult_race.load()
    assert (45222, 98) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert data.s.nunique()[0] == 5
    assert (45222, 1) == data.y.shape
    assert adult_race.disc_feature_groups is not None
    assert "race" not in adult_race.disc_feature_groups
    assert "salary" not in adult_race.disc_feature_groups


def test_load_adult_race_binary():
    """test load adult race binary"""
    data: DataTuple = adult("Race-Binary").load()
    assert (45222, 98) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert data.name == "Adult Race-Binary"


def test_load_compas_race():
    """test load compas race"""
    data: DataTuple = compas("Race").load()
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_adult_race_sex():
    """test load adult race sex"""
    with pytest.raises(AssertionError):
        data: DataTuple = adult("Race-Sex").load()
    # assert (45222, 96) == data.x.shape
    # assert (45222, 1) == data.s.shape
    # assert (45222, 1) == data.y.shape
    # assert data.s.nunique()[0] == 6


def test_load_compas_race_sex():
    """test load compas race sex"""
    data: DataTuple = compas("Race-Sex").load(sens_combination=True)
    assert (6167, 399) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape
    assert data.s.nunique()[0] == 4


def test_load_adult_nationality():
    """test load adult nationality"""
    data: DataTuple = adult("Nationality").load()
    assert (45222, 62) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert data.s.nunique()[0] == 41


def test_race_feature_split():
    """test race feature split"""
    adult_data: Dataset = adult(split="Custom")
    adult_data = replace(
        adult_data,
        sens_attrs=["race_White"],
        s_prefix=["race"],
        class_labels=["salary_>50K"],
        class_label_prefix=["salary"],
    )

    data: DataTuple = adult_data.load()

    assert (45222, 98) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_load_adult_drop_native():
    """test load adult drop native"""
    adult_data = adult("Sex", binarize_nationality=True)
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
    """test load adult education"""
    adult_data = adult("Education")
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
    """test load adult education"""
    adult_data = adult("Education", binarize_nationality=True)
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
    """test additional columns load"""
    data_loc = data_root / "adult.csv.zip"
    data_obj: Dataset = create_data_obj(
        data_loc,
        s_columns=["race_White"],
        y_columns=["salary_>50K"],
        additional_to_drop=["race_Black", "salary_<=50K"],
    )
    data: DataTuple = data_obj.load()

    assert (45222, 102) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_domain_adapt_adult():
    """test domain adapt adult"""
    data: DataTuple = adult().load()
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

    data = adult().load()
    train, test = domain_split(
        datatup=data, tr_cond="education_Masters == 0.", te_cond="education_Masters == 1."
    )
    assert (40194, 101) == train.x.shape
    assert (40194, 1) == train.s.shape
    assert (40194, 1) == train.y.shape

    assert (5028, 101) == test.x.shape
    assert (5028, 1) == test.s.shape
    assert (5028, 1) == test.y.shape

    data = adult().load()
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


def test_celeba():
    """test celeba"""
    celeba_data, _ = celeba(download_dir="non-existent")
    assert celeba_data is None  # data should not be there
    celeba_data, _ = celeba(download_dir="non-existent", check_integrity=False)
    assert celeba_data is not None
    data = celeba_data.load(map_to_binary=True)

    assert celeba_data.name == "CelebA, s=[Male], y=Smiling"

    assert (202599, 39) == data.x.shape
    assert (202599, 1) == data.s.shape
    assert (202599, 1) == data.y.shape
    assert len(data) == len(celeba_data)

    assert data.x["filename"].iloc[0] == "000001.jpg"


def test_celeba_multi_s():
    """test celeba"""
    celeba_data, _ = celeba(
        sens_attr=["Young", "Male"], download_dir="non-existent", check_integrity=False
    )
    assert celeba_data is not None
    data = celeba_data.load(sens_combination=True, map_to_binary=True)

    assert celeba_data.name == "CelebA, s=[Young, Male], y=Smiling"
    assert celeba_data.combination_multipliers == {"Young": 1, "Male": 2}

    assert np.unique(data.s.to_numpy()).tolist() == [0, 1, 2, 3]
    assert data.s.columns[0] == "Young,Male"
    assert (202599, 38) == data.x.shape
    assert (202599, 1) == data.s.shape
    assert (202599, 1) == data.y.shape
    assert len(data) == len(celeba_data)

    assert data.x["filename"].iloc[0] == "000001.jpg"


def test_genfaces():
    """test genfaces"""
    gen_faces, _ = genfaces(download_dir="non-existent")
    assert gen_faces is None  # data should not be there
    gen_faces, _ = genfaces(download_dir="non-existent", check_integrity=False)
    assert gen_faces is not None
    data = gen_faces.load()

    assert gen_faces.name == "GenFaces, s=gender, y=emotion"

    assert (148285, 18) == data.x.shape
    assert (148285, 1) == data.s.shape
    assert (148285, 1) == data.y.shape
    assert len(data) == len(gen_faces)

    assert data.x["filename"].iloc[0] == "5e011b2e7b1b30000702aa59.jpg"
