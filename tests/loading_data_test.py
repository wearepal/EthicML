"""
Test the loading data capability
"""

import pandas as pd

from ethicml.common import ROOT_DIR
from ethicml.data.dataset import Dataset
from ethicml.data.load import load_data, create_data_obj
from ethicml.data import Adult, Compas, Credit, German, Sqf, Toy, NonBinaryToy
from ethicml.algorithms.utils import DataTuple
from ethicml.preprocessing.domain_adaptation import domain_split


def test_can_load_test_data():
    data_loc: str = "{}/data/csvs/toy.csv".format(ROOT_DIR)
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


def test_load_data():
    data: DataTuple = load_data(Toy())
    assert (2000, 2) == data.x.shape
    assert (2000, 1) == data.s.shape
    assert (2000, 1) == data.y.shape


def test_load_non_binary_data():
    data: DataTuple = load_data(NonBinaryToy())
    assert (100, 2) == data.x.shape
    assert (100, 1) == data.s.shape
    assert (100, 1) == data.y.shape


def test_discrete_data():
    data: DataTuple = load_data(Adult())
    assert (48842, 102) == data.x.shape
    assert (48842, 1) == data.s.shape
    assert (48842, 1) == data.y.shape
    assert 97 == len(Adult().discrete_features)
    assert 94 == len(Adult(split='Race').discrete_features)
    assert 58 == len(Adult(split='Nationality').discrete_features)


def test_load_data_as_a_function():
    data_loc: str = "{}/data/csvs/toy.csv".format(ROOT_DIR)
    data_obj: Dataset = create_data_obj(data_loc, s_columns=["s"], y_columns=["y"])
    assert data_obj is not None
    assert data_obj.feature_split['x'] == ['a1', 'a2']
    assert data_obj.feature_split['s'] == ['s']
    assert data_obj.feature_split['y'] == ['y']
    assert data_obj.filename == "toy.csv"


def test_joining_2_load_functions():
    data_loc: str = "{}/data/csvs/toy.csv".format(ROOT_DIR)
    data_obj: Dataset = create_data_obj(data_loc, s_columns=["s"], y_columns=["y"])
    data: DataTuple = load_data(data_obj)
    assert (2000, 2) == data.x.shape
    assert (2000, 1) == data.s.shape
    assert (2000, 1) == data.y.shape


def test_load_adult():
    data: DataTuple = load_data(Adult())
    assert (48842, 102) == data.x.shape
    assert (48842, 1) == data.s.shape
    assert (48842, 1) == data.y.shape


def test_load_compas():
    data: DataTuple = load_data(Compas())
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_sqf():
    data: DataTuple = load_data(Sqf())
    assert (12347, 144) == data.x.shape
    assert (12347, 1) == data.s.shape
    assert (12347, 1) == data.y.shape


def test_load_german():
    data: DataTuple = load_data(German())
    assert (1000, 57) == data.x.shape
    assert (1000, 1) == data.s.shape
    assert (1000, 1) == data.y.shape


def test_load_german_ordered():
    data: DataTuple = load_data(German(), ordered=True)
    assert (1000, 57) == data.x.shape
    assert (1000, 1) == data.s.shape
    assert (1000, 1) == data.y.shape


def test_load_adult_explicitly_sex():
    data: DataTuple = load_data(Adult("Sex"))
    assert (48842, 102) == data.x.shape
    assert (48842, 1) == data.s.shape
    assert (48842, 1) == data.y.shape


def test_load_compas_explicitly_sex():
    data: DataTuple = load_data(Compas("Sex"))
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_compas_feature_length():
    data: DataTuple = load_data(Compas())
    assert (400 == len(Compas().ordered_features['x']))
    assert (395 == len(Compas().discrete_features))
    assert (5 == len(Compas().continuous_features))
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_credit_feature_length():
    data: DataTuple = load_data(Credit())
    assert (26 == len(Credit().ordered_features['x']))
    assert (6 == len(Credit().discrete_features))
    assert (20 == len(Credit().continuous_features))
    assert (30000, 1) == data.s.shape
    assert (30000, 1) == data.y.shape


def test_load_adult_race():
    data: DataTuple = load_data(Adult("Race"))
    assert (48842, 99) == data.x.shape
    assert (48842, 5) == data.s.shape
    assert (48842, 1) == data.y.shape


def test_load_compas_race():
    data: DataTuple = load_data(Compas("Race"))
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_adult_race_sex():
    data: DataTuple = load_data(Adult("Race-Sex"))
    assert (48842, 97) == data.x.shape
    assert (48842, 6) == data.s.shape
    assert (48842, 1) == data.y.shape


def test_load_compas_race_sex():
    data: DataTuple = load_data(Compas("Race-Sex"))
    assert (6167, 399) == data.x.shape
    assert (6167, 2) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_adult_nationality():
    data: DataTuple = load_data(Adult("Nationality"))
    assert (48842, 63) == data.x.shape
    assert (48842, 41) == data.s.shape
    assert (48842, 1) == data.y.shape


def test_race_feature_split():
    adult: Adult = Adult(split="Custom")
    adult.sens_attrs = ["race_White"]
    adult.s_prefix = ["race"]
    adult.class_labels = ["salary_>50K"]
    adult.class_label_prefix = ["salary"]

    data: DataTuple = load_data(adult)

    assert (48842, 99) == data.x.shape
    assert (48842, 1) == data.s.shape
    assert (48842, 1) == data.y.shape


def test_additional_columns_load():
    data_loc: str = "{}/data/csvs/adult.csv".format(ROOT_DIR)
    data_obj: Dataset = create_data_obj(data_loc,
                                        s_columns=["race_White"],
                                        y_columns=["salary_>50K"],
                                        additional_to_drop=[
                                            "race_Black",
                                            "salary_<=50K"
                                        ])
    data: DataTuple = load_data(data_obj)

    assert (48842, 102) == data.x.shape
    assert (48842, 1) == data.s.shape
    assert (48842, 1) == data.y.shape


def test_domain_adapt_adult():
    data: DataTuple = load_data(Adult())
    train, test = domain_split(datatup=data,
                               tr_cond='education_Masters == 0. & education_Doctorate == 0.',
                               te_cond='education_Masters == 1. | education_Doctorate == 1.')
    assert (42340, 102) == train.x.shape
    assert (42340, 1) == train.s.shape
    assert (42340, 1) == train.y.shape

    assert (6502, 102) == test.x.shape
    assert (6502, 1) == test.s.shape
    assert (6502, 1) == test.y.shape

    data = load_data(Adult())
    train, test = domain_split(datatup=data,
                               tr_cond='education_Masters == 0.',
                               te_cond='education_Masters == 1.')
    assert (43528, 102) == train.x.shape
    assert (43528, 1) == train.s.shape
    assert (43528, 1) == train.y.shape

    assert (5314, 102) == test.x.shape
    assert (5314, 1) == test.s.shape
    assert (5314, 1) == test.y.shape

    data = load_data(Adult())
    train, test = domain_split(datatup=data,
                               tr_cond='education_Masters == 0. & education_Doctorate == 0. & education_Bachelors == 0.',
                               te_cond='education_Masters == 1. | education_Doctorate == 1. | education_Bachelors == 1.')
    assert (26290, 102) == train.x.shape
    assert (26290, 1) == train.s.shape
    assert (26290, 1) == train.y.shape

    assert (22552, 102) == test.x.shape
    assert (22552, 1) == test.s.shape
    assert (22552, 1) == test.y.shape
