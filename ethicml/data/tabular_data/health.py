"""Class to describe features of the Heritage Health dataset."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Type

from ..dataset import LegacyDataset
from ..util import flatten_dict

__all__ = ["Health", "HealthSplits"]


class HealthSplits(Enum):
    """Splits for the Health dataset."""

    SEX = "Sex"
    CUSTOM = "Custom"


@dataclass
class Health(LegacyDataset):
    """Heritage Health dataset."""

    split: HealthSplits = HealthSplits.SEX

    Splits: ClassVar[Type[HealthSplits]] = HealthSplits
    """Shorthand for the Enum that defines the splits associated with this class."""

    def __post_init__(self) -> None:
        disc_feature_groups = {
            "age": [
                "age_05",
                "age_15",
                "age_25",
                "age_35",
                "age_45",
                "age_55",
                "age_65",
                "age_75",
                "age_85",
            ],
            "sex": [
                "sexMALE",
                "sexFEMALE",
            ],
            "labNull": ["labNull"],
            "drugNull": ["drugNull"],
            "ClaimsTruncated": ["ClaimsTruncated"],
            "Charlson>0": ["Charlson>0"],
        }
        discrete_features = flatten_dict(disc_feature_groups)

        continuous_features = [
            "no_Claims",
            "no_Providers",
            "no_Vendors",
            "no_PCPs",
            "no_PlaceSvcs",
            "no_Specialities",
            "no_PrimaryConditionGroups",
            "no_ProcedureGroups",
            "PayDelay_max",
            "PayDelay_min",
            "PayDelay_ave",
            "PayDelay_stdev",
            "LOS_max",
            "LOS_min",
            "LOS_ave",
            "LOS_stdev",
            "LOS_TOT_UNKNOWN",
            "LOS_TOT_SUPRESSED",
            "LOS_TOT_KNOWN",
            "dsfs_max",
            "dsfs_min",
            "dsfs_range",
            "dsfs_ave",
            "dsfs_stdev",
            "pcg1",
            "pcg2",
            "pcg3",
            "pcg4",
            "pcg5",
            "pcg6",
            "pcg7",
            "pcg8",
            "pcg9",
            "pcg10",
            "pcg11",
            "pcg12",
            "pcg13",
            "pcg14",
            "pcg15",
            "pcg16",
            "pcg17",
            "pcg18",
            "pcg19",
            "pcg20",
            "pcg21",
            "pcg22",
            "pcg23",
            "pcg24",
            "pcg25",
            "pcg26",
            "pcg27",
            "pcg28",
            "pcg29",
            "pcg30",
            "pcg31",
            "pcg32",
            "pcg33",
            "pcg34",
            "pcg35",
            "pcg36",
            "pcg37",
            "pcg38",
            "pcg39",
            "pcg40",
            "pcg41",
            "pcg42",
            "pcg43",
            "pcg44",
            "pcg45",
            "pcg46",
            "sp1",
            "sp2",
            "sp3",
            "sp4",
            "sp5",
            "sp6",
            "sp7",
            "sp8",
            "sp9",
            "sp10",
            "sp11",
            "sp12",
            "sp13",
            "pg1",
            "pg2",
            "pg3",
            "pg4",
            "pg5",
            "pg6",
            "pg7",
            "pg8",
            "pg9",
            "pg10",
            "pg11",
            "pg12",
            "pg13",
            "pg14",
            "pg15",
            "pg16",
            "pg17",
            "pg18",
            "ps1",
            "ps2",
            "ps3",
            "ps4",
            "ps5",
            "ps6",
            "ps7",
            "ps8",
            "ps9",
            "drugCount_max",
            "drugCount_min",
            "drugCount_ave",
            "drugcount_months",
            "labCount_max",
            "labCount_min",
            "labCount_ave",
            "labcount_months",
        ]

        if self.split is HealthSplits.SEX:
            sens_attr_spec = "sexMALE"
            s_prefix = ["sex"]
            class_label_spec = "Charlson>0"
            class_label_prefix = ["Charlson>0"]
        elif self.split is HealthSplits.CUSTOM:
            sens_attr_spec = ""
            s_prefix = []
            class_label_spec = ""
            class_label_prefix = []
        else:
            raise NotImplementedError
        super().__init__(
            name="Health",
            num_samples=171_067,
            filename_or_path="health.csv.zip",
            features=discrete_features + continuous_features,
            cont_features=continuous_features,
            s_feature_groups=s_prefix,
            sens_attr_spec=sens_attr_spec,
            class_feature_groups=class_label_prefix,
            class_label_spec=class_label_spec,
            discrete_feature_groups=disc_feature_groups,
        )
