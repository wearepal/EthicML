"""
Test the loading data capability
"""
from pathlib import Path
import pandas as pd

from ethicml.common import ROOT_PATH
from ethicml.data import (
    Dataset, load_data, create_data_obj, Adult, Compas, Credit, German, Sqf, Toy, NonBinaryToy)
from ethicml.utility import DataTuple, concat_dt
from ethicml.preprocessing import domain_split, query_dt


def test_can_load_test_data():
    data_loc: Path = ROOT_PATH / "data" / "csvs" / "toy.csv"
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


def test_load_data():
    data: DataTuple = load_data(Toy())
    assert (2000, 2) == data.x.shape
    assert (2000, 1) == data.s.shape
    assert (2000, 1) == data.y.shape
    assert "Toy" == data.name


def test_load_non_binary_data():
    data: DataTuple = load_data(NonBinaryToy())
    assert (100, 2) == data.x.shape
    assert (100, 1) == data.s.shape
    assert (100, 1) == data.y.shape


def test_discrete_data():
    data: DataTuple = load_data(Adult())
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert 96 == len(Adult().discrete_features)
    assert 93 == len(Adult(split='Race').discrete_features)
    assert 57 == len(Adult(split='Nationality').discrete_features)


def test_load_data_as_a_function():
    data_loc: Path = ROOT_PATH / "data" / "csvs" / "toy.csv"
    data_obj: Dataset = create_data_obj(data_loc, s_columns=["s"], y_columns=["y"])
    assert data_obj is not None
    assert data_obj.feature_split['x'] == ['a1', 'a2']
    assert data_obj.feature_split['s'] == ['s']
    assert data_obj.feature_split['y'] == ['y']
    assert data_obj.filename == "toy.csv"


def test_joining_2_load_functions():
    data_loc: Path = ROOT_PATH / "data" / "csvs" / "toy.csv"
    data_obj: Dataset = create_data_obj(data_loc, s_columns=["s"], y_columns=["y"])
    data: DataTuple = load_data(data_obj)
    assert (2000, 2) == data.x.shape
    assert (2000, 1) == data.s.shape
    assert (2000, 1) == data.y.shape


def test_load_adult():
    data: DataTuple = load_data(Adult())
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape
    assert "Adult" == data.name


def test_load_compas():
    data: DataTuple = load_data(Compas())
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape
    assert "Compas" == data.name


def test_load_sqf():
    data: DataTuple = load_data(Sqf())
    assert (12347, 144) == data.x.shape
    assert (12347, 1) == data.s.shape
    assert (12347, 1) == data.y.shape
    assert "SQF" == data.name


def test_load_german():
    data: DataTuple = load_data(German())
    assert (1000, 57) == data.x.shape
    assert (1000, 1) == data.s.shape
    assert (1000, 1) == data.y.shape
    assert "German" == data.name


def test_load_german_ordered():
    data: DataTuple = load_data(German(), ordered=True)
    assert (1000, 57) == data.x.shape
    assert (1000, 1) == data.s.shape
    assert (1000, 1) == data.y.shape


def test_load_adult_explicitly_sex():
    data: DataTuple = load_data(Adult("Sex"))
    assert (45222, 101) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


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
    assert (45222, 98) == data.x.shape
    assert (45222, 5) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_load_compas_race():
    data: DataTuple = load_data(Compas("Race"))
    assert (6167, 400) == data.x.shape
    assert (6167, 1) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_adult_race_sex():
    data: DataTuple = load_data(Adult("Race-Sex"))
    assert (45222, 96) == data.x.shape
    assert (45222, 6) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_load_compas_race_sex():
    data: DataTuple = load_data(Compas("Race-Sex"))
    assert (6167, 399) == data.x.shape
    assert (6167, 2) == data.s.shape
    assert (6167, 1) == data.y.shape


def test_load_adult_nationality():
    data: DataTuple = load_data(Adult("Nationality"))
    assert (45222, 62) == data.x.shape
    assert (45222, 41) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_race_feature_split():
    adult: Adult = Adult(split="Custom")
    adult.sens_attrs = ["race_White"]
    adult.s_prefix = ["race"]
    adult.class_labels = ["salary_>50K"]
    adult.class_label_prefix = ["salary"]

    data: DataTuple = load_data(adult)

    assert (45222, 98) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_additional_columns_load():
    data_loc: Path = ROOT_PATH / "data" / "csvs" / "adult.csv"
    data_obj: Dataset = create_data_obj(data_loc,
                                        s_columns=["race_White"],
                                        y_columns=["salary_>50K"],
                                        additional_to_drop=[
                                            "race_Black",
                                            "salary_<=50K"
                                        ])
    data: DataTuple = load_data(data_obj)

    assert (45222, 102) == data.x.shape
    assert (45222, 1) == data.s.shape
    assert (45222, 1) == data.y.shape


def test_domain_adapt_adult():
    data: DataTuple = load_data(Adult())
    train, test = domain_split(
        datatup=data,
        tr_cond='education_Masters == 0. & education_Doctorate == 0.',
        te_cond='education_Masters == 1. | education_Doctorate == 1.')
    assert (39106, 101) == train.x.shape
    assert (39106, 1) == train.s.shape
    assert (39106, 1) == train.y.shape

    assert (6116, 101) == test.x.shape
    assert (6116, 1) == test.s.shape
    assert (6116, 1) == test.y.shape

    data = load_data(Adult())
    train, test = domain_split(
        datatup=data,
        tr_cond='education_Masters == 0.',
        te_cond='education_Masters == 1.')
    assert (40194, 101) == train.x.shape
    assert (40194, 1) == train.s.shape
    assert (40194, 1) == train.y.shape

    assert (5028, 101) == test.x.shape
    assert (5028, 1) == test.s.shape
    assert (5028, 1) == test.y.shape

    data = load_data(Adult())
    train, test = domain_split(
        datatup=data,
        tr_cond='education_Masters == 0. & education_Doctorate == 0. & education_Bachelors == 0.',
        te_cond='education_Masters == 1. | education_Doctorate == 1. | education_Bachelors == 1.')
    assert (23966, 101) == train.x.shape
    assert (23966, 1) == train.s.shape
    assert (23966, 1) == train.y.shape

    assert (21256, 101) == test.x.shape
    assert (21256, 1) == test.s.shape
    assert (21256, 1) == test.y.shape


def test_query():
    x: pd.DataFrame = pd.DataFrame(columns=['0a', 'b'],
                                   data=[[0, 1],
                                         [2, 3],
                                         [4, 5]])
    s: pd.DataFrame = pd.DataFrame(columns=['c='],
                                   data=[[6],
                                         [7],
                                         [8]])
    y: pd.DataFrame = pd.DataFrame(columns=['d'],
                                   data=[[9],
                                         [10],
                                         [11]])
    data = DataTuple(x=x, s=s, y=y, name='test_data')
    selected = query_dt(data, '_0a == 0 & c_eq_ == 6 & d == 9')
    pd.testing.assert_frame_equal(selected.x, x.head(1))
    pd.testing.assert_frame_equal(selected.s, s.head(1))
    pd.testing.assert_frame_equal(selected.y, y.head(1))


def test_concat():
    x: pd.DataFrame = pd.DataFrame(columns=['a'], data=[[1]])
    s: pd.DataFrame = pd.DataFrame(columns=['b'], data=[[2]])
    y: pd.DataFrame = pd.DataFrame(columns=['c'], data=[[3]])
    data1 = DataTuple(x=x, s=s, y=y, name='test_data')
    x = pd.DataFrame(columns=['a'], data=[[4]])
    s = pd.DataFrame(columns=['b'], data=[[5]])
    y = pd.DataFrame(columns=['c'], data=[[6]])
    data2 = DataTuple(x=x, s=s, y=y, name='test_tuple')
    data3 = concat_dt([data1, data2], axis='index', ignore_index=True)
    pd.testing.assert_frame_equal(data3.x, pd.DataFrame(columns=['a'], data=[[1], [4]]))
    pd.testing.assert_frame_equal(data3.s, pd.DataFrame(columns=['b'], data=[[2], [5]]))
    pd.testing.assert_frame_equal(data3.y, pd.DataFrame(columns=['c'], data=[[3], [6]]))
