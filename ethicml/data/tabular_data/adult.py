"""Class to describe features of the Adult dataset."""
from dataclasses import dataclass
from enum import Enum
from typing import Union

from ..dataset import LoadableDataset
from ..util import LabelSpec, flatten_dict, reduce_feature_group, simple_spec

__all__ = ["Adult", "AdultSplits", "adult"]


class AdultSplits(Enum):
    """Available dataset splits for the Adult dataset."""

    SEX = "Sex"
    EDUCTAION = "Education"
    NATIONALITY = "Nationality"
    RACE = "Race"
    RACE_BINARY = "Race-Binary"
    RACE_SEX = "Race-Sex"
    CUSTOM = "Custom"


def adult(
    split: Union[AdultSplits, str] = "Sex",
    discrete_only: bool = False,
    binarize_nationality: bool = False,
    binarize_race: bool = False,
    invert_s: bool = False,
) -> "Adult":
    """UCI Adult dataset."""
    return Adult(
        split=AdultSplits(split),
        discrete_only=discrete_only,
        binarize_nationality=binarize_nationality,
        binarize_race=binarize_race,
        invert_s=invert_s,
    )


@dataclass
class Adult(LoadableDataset):
    """UCI Adult dataset."""

    split: AdultSplits = AdultSplits.SEX
    binarize_nationality: bool = False
    binarize_race: bool = False

    def __post_init__(self) -> None:
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
            "salary": ["salary_<=50K", "salary_>50K"],
            "sex": ["sex_Female", "sex_Male"],
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
        discrete_features = flatten_dict(disc_feature_groups)

        continuous_features = [
            "age",
            "capital-gain",
            "capital-loss",
            "education-num",
            "hours-per-week",
        ]

        sens_attr_spec: Union[str, LabelSpec]
        if self.split is AdultSplits.SEX:
            sens_attr_spec = "sex_Male"
            s_prefix = ["sex"]
            class_label_spec = "salary_>50K"
            class_label_prefix = ["salary"]
        elif self.split is AdultSplits.RACE:
            sens_attr_spec = simple_spec({"race": disc_feature_groups["race"]})
            s_prefix = ["race"]
            class_label_spec = "salary_>50K"
            class_label_prefix = ["salary"]
        elif self.split is AdultSplits.RACE_BINARY:
            sens_attr_spec = "race_White"
            s_prefix = ["race"]
            class_label_spec = "salary_>50K"
            class_label_prefix = ["salary"]
        elif self.split is AdultSplits.CUSTOM:
            sens_attr_spec = ""
            s_prefix = []
            class_label_spec = ""
            class_label_prefix = []
        elif self.split is AdultSplits.RACE_SEX:
            sens_attr_spec = simple_spec({"sex": ["sex_Male"], "race": disc_feature_groups["race"]})
            s_prefix = ["race", "sex"]
            class_label_spec = "salary_>50K"
            class_label_prefix = ["salary"]
        elif self.split is AdultSplits.NATIONALITY:
            sens = "native-country"
            sens_attr_spec = simple_spec({sens: disc_feature_groups[sens]})
            s_prefix = ["native-country"]
            class_label_spec = "salary_>50K"
            class_label_prefix = ["salary"]
        elif self.split is AdultSplits.EDUCTAION:
            to_keep = ["education_HS-grad", "education_Some-college"]
            remaining_feature_name = "other"
            discrete_features = reduce_feature_group(
                disc_feature_groups=disc_feature_groups,
                feature_group="education",
                to_keep=to_keep,
                remaining_feature_name="_" + remaining_feature_name,
            )
            sens_attr_spec = simple_spec(
                {"education": to_keep + ["education_" + remaining_feature_name]}
            )
            s_prefix = ["education"]
            class_label_spec = "salary_>50K"
            class_label_prefix = ["salary"]
        else:
            raise NotImplementedError

        name = f"Adult {self.split.value}"
        if self.binarize_nationality:
            discrete_features = reduce_feature_group(
                disc_feature_groups=disc_feature_groups,
                feature_group="native-country",
                to_keep=["native-country_United-States"],
                remaining_feature_name="_not_United-States",
            )
            if self.split is AdultSplits.SEX:
                assert len(discrete_features) == 61  # 57 (discrete) features + 4 class labels
            name += ", binary nationality"
        if self.binarize_race:
            discrete_features = reduce_feature_group(
                disc_feature_groups=disc_feature_groups,
                feature_group="race",
                to_keep=["race_White"],
                remaining_feature_name="_not_White",
            )
            if self.split is AdultSplits.SEX and self.binarize_nationality:
                assert len(discrete_features) == 58  # 54 (discrete) features + 4 class labels
            if self.split is AdultSplits.SEX and not self.binarize_nationality:
                assert len(discrete_features) == 97  # 93 (discrete) features + 4 class labels
            name += ", binary race"

        super().__init__(
            name=name,
            num_samples=45222,
            features=discrete_features + continuous_features,
            cont_features=continuous_features,
            sens_attr_spec=sens_attr_spec,
            class_label_spec=class_label_spec,
            filename_or_path="adult.csv.zip",
            s_prefix=s_prefix,
            class_label_prefix=class_label_prefix,
            discrete_only=self.discrete_only,
            discrete_feature_groups=disc_feature_groups,
        )
