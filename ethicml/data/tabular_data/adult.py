"""Class to describe features of the Adult dataset."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, List, Tuple, Type, Union
from typing_extensions import Final

from ranzen import implements

from ..dataset import CSVDatasetDC, LabelSpecsPair
from ..util import (
    DiscFeatureGroup,
    flatten_dict,
    reduce_feature_group,
    simple_spec,
    single_col_spec,
)

__all__ = ["Adult", "AdultSplits"]


class AdultSplits(Enum):
    """Available dataset splits for the Adult dataset."""

    SEX = "Sex"
    EDUCTAION = "Education"
    NATIONALITY = "Nationality"
    RACE = "Race"
    RACE_BINARY = "Race-Binary"
    RACE_SEX = "Race-Sex"


@dataclass
class Adult(CSVDatasetDC):
    """UCI Adult dataset.

    :param discrete_only: If True, continuous features are dropped. (Default: False)
    :param invert_s: If True, the (binary) ``s`` values are inverted. (Default: False)
    :param split: What to use as ``s``. (Default: "Sex")
    :param binarize_nationality: If True, nationality will be USA vs rest. (Default: False)
    :param binarize_race: If True, race will be white vs rest. (Default: False)
    """

    Splits: ClassVar[Type[AdultSplits]] = AdultSplits

    split: AdultSplits = Splits.SEX
    binarize_nationality: bool = False
    binarize_race: bool = False

    @implements(CSVDatasetDC)
    def get_num_samples(self) -> int:
        return 45222

    @implements(CSVDatasetDC)
    def get_filename_or_path(self) -> Union[str, Path]:
        return "adult.csv.zip"

    @implements(CSVDatasetDC)
    def get_name(self) -> str:
        name = f"Adult {self.split.value}"
        if self.binarize_nationality:
            name += ", binary nationality"
        if self.binarize_race:
            name += ", binary race"
        return name

    @implements(CSVDatasetDC)
    def get_label_specs(self) -> Tuple[LabelSpecsPair, List[str]]:
        disc_feature_groups = self.get_disc_feature_groups()
        class_label_spec = single_col_spec("salary_>50K")
        label_feature_groups = ["salary"]
        if self.split is AdultSplits.SEX:
            sens_attr_spec = single_col_spec("sex_Male")
            label_feature_groups += ["sex"]
        elif self.split is AdultSplits.RACE:
            sens_attr_spec = simple_spec({"race": disc_feature_groups["race"]})
            label_feature_groups += ["race"]
        elif self.split is AdultSplits.RACE_BINARY:
            sens_attr_spec = single_col_spec("race_White")
            label_feature_groups += ["race"]
        elif self.split is AdultSplits.RACE_SEX:
            sens_attr_spec = simple_spec({"sex": ["sex_Male"], "race": disc_feature_groups["race"]})
            label_feature_groups += ["sex", "race"]
        elif self.split is AdultSplits.NATIONALITY:
            sens = "native-country"
            sens_attr_spec = simple_spec({sens: disc_feature_groups[sens]})
            label_feature_groups += ["native-country"]
        elif self.split is AdultSplits.EDUCTAION:
            to_keep = ["education_HS-grad", "education_Some-college"]
            remaining_feature_name = "other"

            sens_attr_spec = simple_spec(
                {"education": to_keep + [f"education_{remaining_feature_name}"]}
            )
            label_feature_groups += ["education"]

        else:
            raise NotImplementedError
        return LabelSpecsPair(s=sens_attr_spec, y=class_label_spec), label_feature_groups

    @implements(CSVDatasetDC)
    def get_disc_feature_groups(self) -> DiscFeatureGroup:
        dfgs = DISC_FEATURE_GROUPS
        if self.split is AdultSplits.EDUCTAION:
            to_keep = ["education_HS-grad", "education_Some-college"]
            remaining_feature_name = "other"
            dfgs = reduce_feature_group(
                disc_feature_groups=dfgs,
                feature_group="education",
                to_keep=to_keep,
                remaining_feature_name=f"_{remaining_feature_name}",
            )
        if self.binarize_nationality:
            dfgs = reduce_feature_group(
                disc_feature_groups=dfgs,
                feature_group="native-country",
                to_keep=["native-country_United-States"],
                remaining_feature_name="_not_United-States",
            )
            if self.split is AdultSplits.SEX:
                assert len(flatten_dict(dfgs)) == 61  # 57 (discrete) features + 4 class labels
        if self.binarize_race:
            dfgs = reduce_feature_group(
                disc_feature_groups=dfgs,
                feature_group="race",
                to_keep=["race_White"],
                remaining_feature_name="_not_White",
            )
            if self.split is AdultSplits.SEX and self.binarize_nationality:
                assert len(flatten_dict(dfgs)) == 58  # 54 (discrete) features + 4 class labels
            if self.split is AdultSplits.SEX and not self.binarize_nationality:
                assert len(flatten_dict(dfgs)) == 97  # 93 (discrete) features + 4 class labels
        return dfgs

    @implements(CSVDatasetDC)
    def get_unfiltered_continuous(self) -> List[str]:
        feats = [
            "age",
            "capital-gain",
            "capital-loss",
            # "education-num",
            "hours-per-week",
        ]
        if self.split is not AdultSplits.EDUCTAION:
            feats.append("education-num")
        return feats


DISC_FEATURE_GROUPS: Final = {
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
