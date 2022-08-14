"""Class to describe features of the German dataset."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Type

from ..dataset import LegacyDataset
from ..util import flatten_dict

__all__ = ["German", "GermanSplits"]


class GermanSplits(Enum):
    """Splits for the German dataset."""

    SEX = "Sex"
    CUSTOM = "Custom"


@dataclass
class German(LegacyDataset):
    """German credit dataset."""

    split: GermanSplits = GermanSplits.SEX

    Splits: ClassVar[Type[GermanSplits]] = GermanSplits
    """Shorthand for the Enum that defines the splits associated with this class."""

    def __post_init__(self) -> None:
        disc_feature_groups = {
            "sex": ["sex"],
            "age": ["age"],
            "credit-label": ["credit-label"],
            "status": ["status_A11", "status_A12", "status_A13", "status_A14"],
            "credit-history": [
                "credit-history_A30",
                "credit-history_A31",
                "credit-history_A32",
                "credit-history_A33",
                "credit-history_A34",
            ],
            "purpose": [
                "purpose_A40",
                "purpose_A41",
                "purpose_A410",
                "purpose_A42",
                "purpose_A43",
                "purpose_A44",
                "purpose_A45",
                "purpose_A46",
                "purpose_A48",
                "purpose_A49",
            ],
            "savings": [
                "savings_A61",
                "savings_A62",
                "savings_A63",
                "savings_A64",
                "savings_A65",
            ],
            "employment": [
                "employment_A71",
                "employment_A72",
                "employment_A73",
                "employment_A74",
                "employment_A75",
            ],
            "other-debtors": [
                "other-debtors_A101",
                "other-debtors_A102",
                "other-debtors_A103",
            ],
            "property": [
                "property_A121",
                "property_A122",
                "property_A123",
                "property_A124",
            ],
            "installment-plans": [
                "installment-plans_A141",
                "installment-plans_A142",
                "installment-plans_A143",
            ],
            "housing": [
                "housing_A151",
                "housing_A152",
                "housing_A153",
            ],
            "skill-level": [
                "skill-level_A171",
                "skill-level_A172",
                "skill-level_A173",
                "skill-level_A174",
            ],
            "telephone": [
                "telephone_A191",
                "telephone_A192",
            ],
            "foreign-worker": [
                "foreign-worker_A201",
                "foreign-worker_A202",
            ],
        }
        discrete_features = flatten_dict(disc_feature_groups)

        continuous_features = [
            "month",
            "credit-amount",
            "investment-as-income-percentage",
            "residence-since",
            "number-of-credits",
            "people-liable-for",
        ]

        if self.split is GermanSplits.SEX:
            sens_attr_spec = "sex"
            s_prefix = ["sex"]
            class_label_spec = "credit-label"
            class_label_prefix = ["credit-label"]
        elif self.split is GermanSplits.CUSTOM:
            sens_attr_spec = ""
            s_prefix = []
            class_label_spec = ""
            class_label_prefix = []
        else:
            raise NotImplementedError

        super().__init__(
            name=f"German {self.split.value}",
            num_samples=1_000,
            filename_or_path="german.csv",
            features=discrete_features + continuous_features,
            cont_features=continuous_features,
            s_feature_groups=s_prefix,
            sens_attr_spec=sens_attr_spec,
            class_feature_groups=class_label_prefix,
            class_label_spec=class_label_spec,
            discrete_feature_groups=disc_feature_groups,
        )
