"""Class to describe features of the Adult dataset."""
from typing import Dict, List
from warnings import warn

from typing_extensions import Literal

from ..dataset import Dataset

__all__ = ["Adult", "adult"]


def Adult(  # pylint: disable=invalid-name
    split: Literal["Sex", "Race", "Race-Binary", "Race-Sex", "Custom", "Nationality"] = "Sex",
    discrete_only: bool = False,
    binarize_nationality: bool = False,
) -> Dataset:
    """UCI Adult dataset."""
    warn("The Adult class is deprecated. Use the function instead.", DeprecationWarning)
    return adult(split, discrete_only, binarize_nationality)


def adult(
    split: Literal["Sex", "Race", "Race-Binary", "Race-Sex", "Custom", "Nationality"] = "Sex",
    discrete_only: bool = False,
    binarize_nationality: bool = False,
) -> Dataset:
    """UCI Adult dataset."""
    if True:  # pylint: disable=using-constant-test
        disc_feature_groups = {
            "education": [
                "education_10th",
                "education_11th",
                "education_12th",
                "education_1st-4th",
                "education_5th-6th",
                "education_7th-8th",
                "education_9th",
                "education_Assoc-acdm",
                "education_Assoc-voc",
                "education_Bachelors",
                "education_Doctorate",
                "education_HS-grad",
                "education_Masters",
                "education_Preschool",
                "education_Prof-school",
                "education_Some-college",
            ],
            "marital-status": [
                "marital-status_Divorced",
                "marital-status_Married-AF-spouse",
                "marital-status_Married-civ-spouse",
                "marital-status_Married-spouse-absent",
                "marital-status_Never-married",
                "marital-status_Separated",
                "marital-status_Widowed",
            ],
            "native-country": [
                "native-country_Cambodia",
                "native-country_Canada",
                "native-country_China",
                "native-country_Columbia",
                "native-country_Cuba",
                "native-country_Dominican-Republic",
                "native-country_Ecuador",
                "native-country_El-Salvador",
                "native-country_England",
                "native-country_France",
                "native-country_Germany",
                "native-country_Greece",
                "native-country_Guatemala",
                "native-country_Haiti",
                "native-country_Holand-Netherlands",
                "native-country_Honduras",
                "native-country_Hong",
                "native-country_Hungary",
                "native-country_India",
                "native-country_Iran",
                "native-country_Ireland",
                "native-country_Italy",
                "native-country_Jamaica",
                "native-country_Japan",
                "native-country_Laos",
                "native-country_Mexico",
                "native-country_Nicaragua",
                "native-country_Outlying-US(Guam-USVI-etc)",
                "native-country_Peru",
                "native-country_Philippines",
                "native-country_Poland",
                "native-country_Portugal",
                "native-country_Puerto-Rico",
                "native-country_Scotland",
                "native-country_South",
                "native-country_Taiwan",
                "native-country_Thailand",
                "native-country_Trinadad&Tobago",
                "native-country_United-States",
                "native-country_Vietnam",
                "native-country_Yugoslavia",
            ],
            "occupation": [
                "occupation_Adm-clerical",
                "occupation_Armed-Forces",
                "occupation_Craft-repair",
                "occupation_Exec-managerial",
                "occupation_Farming-fishing",
                "occupation_Handlers-cleaners",
                "occupation_Machine-op-inspct",
                "occupation_Other-service",
                "occupation_Priv-house-serv",
                "occupation_Prof-specialty",
                "occupation_Protective-serv",
                "occupation_Sales",
                "occupation_Tech-support",
                "occupation_Transport-moving",
            ],
            "race": [
                "race_Amer-Indian-Eskimo",
                "race_Asian-Pac-Islander",
                "race_Black",
                "race_Other",
                "race_White",
            ],
            "relationship": [
                "relationship_Husband",
                "relationship_Not-in-family",
                "relationship_Other-relative",
                "relationship_Own-child",
                "relationship_Unmarried",
                "relationship_Wife",
            ],
            "salary": ["salary_<=50K", "salary_>50K",],
            "sex": ["sex_Female", "sex_Male",],
            "workclass": [
                "workclass_Federal-gov",
                "workclass_Local-gov",
                "workclass_Private",
                "workclass_Self-emp-inc",
                "workclass_Self-emp-not-inc",
                "workclass_State-gov",
                "workclass_Without-pay",
            ],
        }

        discrete_features: List[str] = []
        for group in disc_feature_groups.values():
            discrete_features += group

        continuous_features = [
            "age",
            "capital-gain",
            "capital-loss",
            "education-num",
            "hours-per-week",
        ]

        if split == "Sex":
            sens_attrs = ["sex_Male"]
            s_prefix = ["sex"]
            class_labels = ["salary_>50K"]
            class_label_prefix = ["salary"]
        elif split == "Race":
            sens_attrs = [
                "race_Amer-Indian-Eskimo",
                "race_Asian-Pac-Islander",
                "race_Black",
                "race_Other",
                "race_White",
            ]
            s_prefix = ["race"]
            class_labels = ["salary_>50K"]
            class_label_prefix = ["salary"]
        elif split == "Race-Binary":
            sens_attrs = ["race_White"]
            s_prefix = ["race"]
            class_labels = ["salary_>50K"]
            class_label_prefix = ["salary"]
        elif split == "Custom":
            sens_attrs = []
            s_prefix = []
            class_labels = []
            class_label_prefix = []
        elif split == "Race-Sex":
            sens_attrs = [
                "sex_Male",
                "race_Amer-Indian-Eskimo",
                "race_Asian-Pac-Islander",
                "race_Black",
                "race_Other",
                "race_White",
            ]
            s_prefix = ["race", "sex"]
            class_labels = ["salary_>50K"]
            class_label_prefix = ["salary"]
        elif split == "Nationality":
            sens_attrs = [
                "native-country_Cambodia",
                "native-country_Canada",
                "native-country_China",
                "native-country_Columbia",
                "native-country_Cuba",
                "native-country_Dominican-Republic",
                "native-country_Ecuador",
                "native-country_El-Salvador",
                "native-country_England",
                "native-country_France",
                "native-country_Germany",
                "native-country_Greece",
                "native-country_Guatemala",
                "native-country_Haiti",
                "native-country_Holand-Netherlands",
                "native-country_Honduras",
                "native-country_Hong",
                "native-country_Hungary",
                "native-country_India",
                "native-country_Iran",
                "native-country_Ireland",
                "native-country_Italy",
                "native-country_Jamaica",
                "native-country_Japan",
                "native-country_Laos",
                "native-country_Mexico",
                "native-country_Nicaragua",
                "native-country_Outlying-US(Guam-USVI-etc)",
                "native-country_Peru",
                "native-country_Philippines",
                "native-country_Poland",
                "native-country_Portugal",
                "native-country_Puerto-Rico",
                "native-country_Scotland",
                "native-country_South",
                "native-country_Taiwan",
                "native-country_Thailand",
                "native-country_Trinadad&Tobago",
                "native-country_United-States",
                "native-country_Vietnam",
                "native-country_Yugoslavia",
            ]
            s_prefix = ["native-country"]
            class_labels = ["salary_>50K"]
            class_label_prefix = ["salary"]
        else:
            raise NotImplementedError

        name = f"Adult {split}"
        if binarize_nationality:
            discrete_features = _drop_native(disc_feature_groups)
            name += ", binary nationality"

    return Dataset(
        name=name,
        num_samples=45222,
        features=discrete_features + continuous_features,
        cont_features=continuous_features,
        sens_attrs=sens_attrs,
        class_labels=class_labels,
        filename_or_path="adult.csv.zip",
        s_prefix=s_prefix,
        class_label_prefix=class_label_prefix,
        discrete_only=discrete_only,
        disc_feature_groups=disc_feature_groups,
    )


def _drop_native(disc_feature_groups: Dict[str, List[str]]) -> List[str]:
    """Drop all features that encode the native country except the one for the US."""
    # first set the native_country feature group to just the value that we want to keep
    disc_feature_groups["native-country"] = ["native-country_United-States"]
    # then, regenerate the list of discrete features; just like it's done in the constructor
    discrete_features: List[str] = []
    for group in disc_feature_groups.values():
        discrete_features += group
    assert len(discrete_features) == 60  # 56 (discrete) input features + 4 label features
    # then add a new dummy feature to the feature group. `load_data()` will create this for us
    disc_feature_groups["native-country"].append("native-country_not_United-States")
    return discrete_features
