"""Class to describe features of the Adult dataset."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Type

from ..dataset import LegacyDataset
from ..util import LabelSpec, flatten_dict

__all__ = ["Nursery", "NurserySplits"]


class NurserySplits(Enum):
    """Available dataset splits for the Adult dataset."""

    FINANCE = "Finance"
    CUSTOM = "Custom"


@dataclass
class Nursery(LegacyDataset):
    """UCI Adult dataset.

    :param discrete_only: If True, continuous features are dropped. (Default: False)
    :param invert_s: If True, the (binary) ``s`` values are inverted. (Default: False)
    :param split: What to use as ``s``. (Default: "Sex")
    :param binarize_nationality: If True, nationality will be USA vs rest. (Default: False)
    :param binarize_race: If True, race will be white vs rest. (Default: False)
    """

    Splits: ClassVar[Type[NurserySplits]] = NurserySplits

    split: NurserySplits = Splits.FINANCE

    def __post_init__(self) -> None:
        disc_feature_groups = {
            "parents": ["parents_usual", "parents_pretentious", "parents_great_pret"],
            "has_nurs": [
                "has_nurs_proper",
                "has_nurs_less_proper",
                "has_nurs_improper",
                "has_nurs_critical",
                "has_nurs_very_crit",
            ],
            "form": ["form_complete", "form_completed", "form_foster", "form_incomplete"],
            "housing": ["housing_convenient", "housing_less_conv", "housing_critical"],
            "health": ["health_not_recom", "health_priority", "health_recommended"],
            "social": ["social_non-prob", "social_slightly_prob", "social_problematic"],
            "class": [
                "class_not_recom",
                "class_recommend",
                "class_very_recom",
                "class_priority",
                "class_spec_prior",
            ],
            "finance": ["finance_conv", "finance_inconv"],
        }
        discrete_features = flatten_dict(disc_feature_groups)

        continuous_features = ["children"]

        sens_attr_spec: str | LabelSpec
        if self.split is NurserySplits.FINANCE:
            sens_attr_spec = "finance_conv"
            s_prefix = ["finance"]
            class_label_spec = "class_not_recom"
            class_label_prefix = ["class"]
        else:
            raise NotImplementedError

        name = f"Nursery {self.split.value}"

        super().__init__(
            name=name,
            num_samples=12960,
            features=discrete_features + continuous_features,
            cont_features=continuous_features,
            sens_attr_spec=sens_attr_spec,
            class_label_spec=class_label_spec,
            filename_or_path="nursery.csv.zip",
            s_feature_groups=s_prefix,
            class_feature_groups=class_label_prefix,
            discrete_feature_groups=disc_feature_groups,
        )
