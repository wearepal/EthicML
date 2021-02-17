"""Test the loading data capability."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ethicml as em
from ethicml import DataTuple


def test_can_load_test_data(data_root: Path):
    """Test whether we can load test data."""
    data_loc = data_root / "toy.csv"
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert data is not None


@pytest.mark.parametrize(
    "dataset,samples,x_features,discrete_features,s_features,num_sens,y_features,num_labels,name",
    [
        (em.adult(), 45222, 101, 96, 1, 2, 1, 2, "Adult Sex"),
        (
            em.adult("Sex", binarize_nationality=True),
            45222,
            62,
            57,
            1,
            2,
            1,
            2,
            "Adult Sex, binary nationality",
        ),
        (
            em.adult("Sex", binarize_race=True),
            45222,
            98,
            93,
            1,
            2,
            1,
            2,
            "Adult Sex, binary race",
        ),
        (
            em.adult("Sex", binarize_nationality=True, binarize_race=True),
            45222,
            59,
            54,
            1,
            2,
            1,
            2,
            "Adult Sex, binary nationality, binary race",
        ),
        (em.adult(split="Sex"), 45_222, 101, 96, 1, 2, 1, 2, "Adult Sex"),
        (em.adult(split="Race"), 45_222, 98, 93, 1, 5, 1, 2, "Adult Race"),
        (em.adult(split="Race-Binary"), 45_222, 98, 93, 1, 2, 1, 2, "Adult Race-Binary"),
        (em.adult(split="Nationality"), 45_222, 62, 57, 1, 41, 1, 2, "Adult Nationality"),
        (em.adult(split="Education"), 45_222, 86, 82, 1, 3, 1, 2, "Adult Education"),
        (em.compas(), 6_167, 400, 395, 1, 2, 1, 2, "Compas Sex"),
        (em.compas(split="Sex"), 6_167, 400, 395, 1, 2, 1, 2, "Compas Sex"),
        (em.compas(split="Race"), 6_167, 400, 395, 1, 2, 1, 2, "Compas Race"),
        (em.compas(split="Race-Sex"), 6_167, 399, 394, 1, 4, 1, 2, "Compas Race-Sex"),
        (em.credit(), 30_000, 26, 6, 1, 2, 1, 2, "Credit Sex"),
        (em.credit(split="Sex"), 30_000, 26, 6, 1, 2, 1, 2, "Credit Sex"),
        (em.crime(), 1_993, 136, 46, 1, 2, 1, 2, "Crime Race-Binary"),
        (em.crime(split="Race-Binary"), 1_993, 136, 46, 1, 2, 1, 2, "Crime Race-Binary"),
        (em.german(), 1_000, 57, 50, 1, 2, 1, 2, "German Sex"),
        (em.german(split="Sex"), 1_000, 57, 50, 1, 2, 1, 2, "German Sex"),
        (em.health(), 171_067, 130, 12, 1, 2, 1, 2, "Health"),
        (em.health(split="Sex"), 171_067, 130, 12, 1, 2, 1, 2, "Health"),
        (em.lipton(), 2_000, 2, 0, 1, 2, 1, 2, "Lipton"),
        (em.nonbinary_toy(), 100, 2, 0, 1, 2, 1, 5, "NonBinaryToy"),
        (em.sqf(), 12_347, 145, 139, 1, 2, 1, 2, "SQF Sex"),
        (em.sqf(split="Sex"), 12_347, 145, 139, 1, 2, 1, 2, "SQF Sex"),
        (em.sqf(split="Race"), 12_347, 145, 139, 1, 2, 1, 2, "SQF Race"),
        (em.sqf(split="Race-Sex"), 12_347, 144, 138, 1, 4, 1, 2, "SQF Race-Sex"),
        (em.toy(), 400, 10, 8, 1, 2, 1, 2, "Toy"),
    ],
)
def test_data_shape(
    dataset,
    samples,
    x_features,
    discrete_features,
    s_features,
    num_sens,
    y_features,
    num_labels,
    name,
):
    """Test loading data."""
    data: DataTuple = dataset.load()
    assert (samples, x_features) == data.x.shape
    assert (samples, s_features) == data.s.shape
    assert (samples, y_features) == data.y.shape

    assert len(dataset.ordered_features["x"]) == x_features
    assert len(dataset.discrete_features) == discrete_features
    assert len(dataset.continuous_features) == (x_features - discrete_features)

    assert data.s.nunique()[0] == num_sens
    assert data.y.nunique()[0] == num_labels

    assert data.name == name

    data: DataTuple = dataset.load(ordered=True)
    assert (samples, x_features) == data.x.shape
    assert (samples, s_features) == data.s.shape
    assert (samples, y_features) == data.y.shape


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

    assert data.s.nunique()[0] == 2
    assert data.y.nunique()[0] == 2

    if fair:
        assert data.name == f"Synthetic - Scenario {scenario}, target {target} fair"
    else:
        assert data.name == f"Synthetic - Scenario {scenario}, target {target}"

    data: DataTuple = dataset.load(ordered=True)
    assert (samples, 4) == data.x.shape
    assert (samples, 1) == data.s.shape
    assert (samples, 1) == data.y.shape


@pytest.mark.parametrize("dataset", [em.Adult, em.Compas, em.Toy])
def test_deprecation_warning(dataset):
    """Test loading data."""
    with pytest.deprecated_call():  # assert that this gives a deprecation warning
        data: DataTuple = dataset().load()


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
    assert data.s.nunique()[0] == 5
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
    assert data.s.nunique()[0] == 2 * 5
    assert (45222, 1) == data.y.shape
    assert adult_race_sex.disc_feature_groups is not None
    assert "race" not in adult_race_sex.disc_feature_groups
    assert "sex" not in adult_race_sex.disc_feature_groups
    assert "salary" not in adult_race_sex.disc_feature_groups


def test_race_feature_split():
    """Test race feature split."""
    adult_data: em.Dataset = em.adult(split="Custom")
    adult_data.sens_attr_spec = "race_White"
    adult_data.s_prefix = ["race"]
    adult_data.class_label_spec = "salary_>50K"
    adult_data.class_label_prefix = ["salary"]

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


def test_celeba():
    """Test celeba."""
    celeba_data, _ = em.celeba(download_dir="non-existent")
    assert celeba_data is None  # data should not be there
    celeba_data, _ = em.celeba(download_dir="non-existent", check_integrity=False)
    assert celeba_data is not None
    data = celeba_data.load()

    assert celeba_data.name == "CelebA, s=Male, y=Smiling"

    assert (202599, 39) == data.x.shape
    assert (202599, 1) == data.s.shape
    assert (202599, 1) == data.y.shape
    assert len(data) == len(celeba_data)

    assert data.x["filename"].iloc[0] == "000001.jpg"


def test_celeba_multi_s():
    """Test celeba w/ multi S."""
    sens_spec = dict(em.simple_spec({"Age": ["Young"], "Gender": ["Male"]}))
    celeba_data, _ = em.celeba(
        sens_attr=sens_spec, download_dir="non-existent", check_integrity=False
    )
    assert celeba_data is not None
    data = celeba_data.load()

    assert celeba_data.name == "CelebA, s=[Age, Gender], y=Smiling"

    assert np.unique(data.s.to_numpy()).tolist() == [0, 1, 2, 3]
    assert data.s.columns[0] == "Age,Gender"
    assert (202599, 38) == data.x.shape
    assert (202599, 1) == data.s.shape
    assert (202599, 1) == data.y.shape
    assert len(data) == len(celeba_data)

    assert data.x["filename"].iloc[0] == "000001.jpg"


@pytest.mark.usefixtures("simulate_no_torch")
def test_celeba_no_torch():
    """Test celeba."""
    with pytest.raises(RuntimeError, match="Need torchvision to download data."):
        em.celeba(download_dir="some_dir", download=True)


def test_genfaces():
    """Test genfaces."""
    gen_faces, _ = em.genfaces(download_dir="non-existent")
    assert gen_faces is None  # data should not be there
    gen_faces, _ = em.genfaces(download_dir="non-existent", check_integrity=False)
    assert gen_faces is not None
    data = gen_faces.load()

    assert gen_faces.name == "GenFaces, s=gender, y=emotion"

    assert (148285, 18) == data.x.shape
    assert (148285, 1) == data.s.shape
    assert (148285, 1) == data.y.shape
    assert len(data) == len(gen_faces)

    assert data.x["filename"].iloc[0] == "5e011b2e7b1b30000702aa59.jpg"


def test_genfaces_multi_s():
    """Test genfaces."""
    gen_faces, _ = em.genfaces(
        label="hair_color",
        sens_attr="eye_color",
        download_dir="non-existent",
        check_integrity=False,
    )
    assert gen_faces is not None
    data = gen_faces.load()

    assert gen_faces.name == "GenFaces, s=eye_color, y=hair_color"

    incomplete_entries = 11_003  # entries that are missing either s or y
    assert (148_285 - incomplete_entries, 11) == data.x.shape
    assert (148_285 - incomplete_entries, 1) == data.s.shape
    assert (148_285 - incomplete_entries, 1) == data.y.shape
    assert len(data) == len(gen_faces) - incomplete_entries

    assert data.x["filename"].iloc[0] == "5e011b2e7b1b30000702aa59.jpg"


def test_expand_s():
    """Test expanding s."""
    data = em.Dataset(
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
