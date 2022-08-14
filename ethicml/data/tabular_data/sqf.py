"""Class to describe features of the SQF dataset."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Type

from ..dataset import LegacyDataset
from ..util import LabelSpec, flatten_dict, spec_from_binary_cols

__all__ = ["Sqf", "SqfSplits"]


class SqfSplits(Enum):
    """Splits for the SQF dataset."""

    SEX = "Sex"
    RACE = "Race"
    RACE_SEX = "Race-Sex"
    CUSTOM = "Custom"


@dataclass
class Sqf(LegacyDataset):
    """Stop, question and frisk dataset.

    This data is from the 2016, source: http://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page
    """

    split: SqfSplits = SqfSplits.SEX

    Splits: ClassVar[Type[SqfSplits]] = SqfSplits
    """Shorthand for the Enum that defines the splits associated with this class."""

    def __post_init__(self) -> None:
        disc_feature_groups = {
            "sex": ["sex"],
            "race": ["race"],
            "weapon": ["weapon"],
            "typeofid": ["typeofid_O", "typeofid_P", "typeofid_R", "typeofid_V"],
            "explnstp": ["explnstp_N", "explnstp_Y"],
            "othpers": ["othpers_N", "othpers_Y"],
            "arstmade": ["arstmade_N", "arstmade_Y"],
            "offunif": ["offunif_N", "offunif_Y"],
            "frisked": ["frisked_N", "frisked_Y"],
            "searched": ["searched_N", "searched_Y"],
            "pf-hands": ["pf-hands_N", "pf-hands_Y"],
            "pf-wall": ["pf-wall_N", "pf-wall_Y"],
            "pf-grnd": ["pf-grnd_N", "pf-grnd_Y"],
            "pf-drweb": ["pf-drwep_N", "pf-drwep_Y"],
            "pf-ptwep": ["pf-ptwep_N", "pf-ptwep_Y"],
            "pf-baton": ["pf-baton_N", "pf-baton_Y"],
            "pf-hcuff": ["pf-hcuff_N", "pf-hcuff_Y"],
            "pf-pepsp": ["pf-pepsp_N", "pf-pepsp_Y"],
            "pf-other": ["pf-other_N", "pf-other_Y"],
            "radio": ["radio_N", "radio_Y"],
            "ac-rept": ["ac-rept_N", "ac-rept_Y"],
            "ac-inves": ["ac-inves_N", "ac-inves_Y"],
            "rf-vcrim": ["rf-vcrim_N", "rf-vcrim_Y"],
            "rf-othsw": ["rf-othsw_N", "rf-othsw_Y"],
            "ac-proxm": ["ac-proxm_N", "ac-proxm_Y"],
            "rf-attir": ["rf-attir_N", "rf-attir_Y"],
            "cs-objcs": ["cs-objcs_N", "cs-objcs_Y"],
            "cs-descr": ["cs-descr_N", "cs-descr_Y"],
            "cs-casng": ["cs-casng_N", "cs-casng_Y"],
            "cs-lkout": ["cs-lkout_N", "cs-lkout_Y"],
            "rf-vacnt": ["rf-vcact_N", "rf-vcact_Y"],
            "cs-cloth": ["cs-cloth_N", "cs-cloth_Y"],
            "cs-drgtr": ["cs-drgtr_N", "cs-drgtr_Y"],
            "ac-evasv": ["ac-evasv_N", "ac-evasv_Y"],
            "ac-assos": ["ac-assoc_N", "ac-assoc_Y"],
            "cs-furtv": ["cs-furtv_N", "cs-furtv_Y"],
            "rf-rfcmp": ["rf-rfcmp_N", "rf-rfcmp_Y"],
            "ac-cgdir": ["ac-cgdir_N", "ac-cgdir_Y"],
            "rf-verbl": ["rf-verbl_N", "rf-verbl_Y"],
            "cs-vcrim": ["cs-vcrim_N", "cs-vcrim_Y"],
            "cs-bulge": ["cs-bulge_N", "cs-bulge_Y"],
            "cs-other": ["cs-other_N", "cs-other_Y"],
            "ac-incid": ["ac-incid_N", "ac-incid_Y"],
            "ac-time": ["ac-time_N", "ac-time_Y"],
            "rf-knowl": ["rf-knowl_N", "rf-knowl_Y"],
            "ac-stsnd": ["ac-stsnd_N", "ac-stsnd_Y"],
            "ac-other": ["ac-other_N", "ac-other_Y"],
            "sb-hdobj": ["sb-hdobj_N", "sb-hdobj_Y"],
            "sb-outln": ["sb-outln_N", "sb-outln_Y"],
            "sb-admis": ["sb-admis_N", "sb-admis_Y"],
            "sb-other": ["sb-other_N", "sb-other_Y"],
            "rf-furt": ["rf-furt_N", "rf-furt_Y"],
            "rf-bulg": ["rf-bulg_N", "rf-bulg_Y"],
            "offverb": ["offverb_ ", "offverb_V"],
            "offshld": ["offshld_ ", "offshld_S"],
            "forceuse": [
                "forceuse_ ",
                "forceuse_DO",
                "forceuse_DS",
                "forceuse_OR",
                "forceuse_OT",
                "forceuse_SF",
                "forceuse_SW",
            ],
            "haircolr": [
                "haircolr_BA",
                "haircolr_BK",
                "haircolr_BL",
                "haircolr_BR",
                "haircolr_DY",
                "haircolr_GY",
                "haircolr_RA",
                "haircolr_SN",
                "haircolr_SP",
                "haircolr_XX",
                "haircolr_ZZ",
            ],
            "eyecolor": [
                "eyecolor_BK",
                "eyecolor_BL",
                "eyecolor_BR",
                "eyecolor_DF",
                "eyecolor_GR",
                "eyecolor_GY",
                "eyecolor_HA",
                "eyecolor_XX",
                "eyecolor_Z",
            ],
            "build": ["build_H", "build_M", "build_T", "build_U", "build_Z"],
        }
        discrete_features = flatten_dict(disc_feature_groups)

        continuous_features = ["perstop", "ht_feet", "age", "ht_inch", "perobs", "weight"]

        sens_attr_spec: str | LabelSpec
        if self.split is SqfSplits.SEX:
            sens_attr_spec = "sex"
            s_prefix = ["sex"]
            class_label_spec = "weapon"
            class_label_prefix = ["weapon"]
        elif self.split is SqfSplits.RACE:
            sens_attr_spec = "race"
            s_prefix = ["race"]
            class_label_spec = "weapon"
            class_label_prefix = ["weapon"]
        elif self.split is SqfSplits.RACE_SEX:
            sens_attr_spec = spec_from_binary_cols({"sex": ["sex"], "race": ["race"]})
            s_prefix = ["race", "sex"]
            class_label_spec = "weapon"
            class_label_prefix = ["weapon"]
        elif self.split is SqfSplits.CUSTOM:
            sens_attr_spec = ""
            s_prefix = []
            class_label_spec = ""
            class_label_prefix = []
        else:
            raise NotImplementedError
        super().__init__(
            name=f"SQF {self.split.value}",
            num_samples=12347,
            filename_or_path="sqf.csv",
            features=discrete_features + continuous_features,
            cont_features=continuous_features,
            s_feature_groups=s_prefix,
            sens_attr_spec=sens_attr_spec,
            class_feature_groups=class_label_prefix,
            class_label_spec=class_label_spec,
            discrete_feature_groups=disc_feature_groups,
        )
