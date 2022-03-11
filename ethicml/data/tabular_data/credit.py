"""Class to describe features of the UCI Credit dataset."""
from dataclasses import dataclass
from enum import Enum
from typing import Union

from ..dataset import LoadableDataset
from ..util import flatten_dict

__all__ = ["credit", "Credit", "CreditSplits"]


class CreditSplits(Enum):
    """Splits for the Credit dataset."""

    SEX = "Sex"
    CUSTOM = "Custom"


def credit(
    split: Union[CreditSplits, str] = "Sex",
    discrete_only: bool = False,
    invert_s: bool = False,
) -> "Credit":
    """UCI Credit Card dataset."""
    return Credit(split=CreditSplits(split), discrete_only=discrete_only, invert_s=invert_s)


@dataclass
class Credit(LoadableDataset):
    """UCI Credit Card dataset."""

    split: CreditSplits = CreditSplits.SEX

    def __post_init__(self) -> None:
        disc_feature_groups = {
            "SEX": ["SEX"],
            "EDUCATION": [
                "EDUCATION_1",
                "EDUCATION_2",
                "EDUCATION_3",
                "EDUCATION_4",
                "EDUCATION_5",
                "EDUCATION_6",
            ],
            "MARRIAGE": [
                "MARRIAGE_1",
                "MARRIAGE_2",
                "MARRIAGE_3",
            ],
            "default-payment-next-month": ["default-payment-next-month"],
        }
        disc_features = flatten_dict(disc_feature_groups)

        continuous_features = [
            "LIMIT_BAL",
            "AGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ]

        if self.split is CreditSplits.SEX:
            sens_attr_spec = "SEX"
            s_prefix = ["SEX"]
            class_label_spec = "default-payment-next-month"
            class_label_prefix = ["default-payment-next-month"]
        elif self.split is CreditSplits.CUSTOM:
            sens_attr_spec = ""
            s_prefix = []
            class_label_spec = ""
            class_label_prefix = []
        else:
            raise NotImplementedError

        super().__init__(
            name=f"Credit {self.split.value}",
            num_samples=30000,
            filename_or_path="UCI_Credit_Card.csv",
            features=disc_features + continuous_features,
            cont_features=continuous_features,
            s_prefix=s_prefix,
            sens_attr_spec=sens_attr_spec,
            class_label_prefix=class_label_prefix,
            class_label_spec=class_label_spec,
            discrete_only=self.discrete_only,
            discrete_feature_groups=disc_feature_groups,
        )
