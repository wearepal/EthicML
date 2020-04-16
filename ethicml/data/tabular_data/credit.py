"""Class to describe features of the Adult dataset."""
from ..dataset import Dataset

__all__ = ["credit"]


def credit(split: str = "Sex", discrete_only: bool = False) -> Dataset:
    """UCL Credit Card dataset."""
    if True:  # pylint: disable=using-constant-test
        features = [
            "LIMIT_BAL",
            "SEX",
            "EDUCATION_1",
            "EDUCATION_2",
            "EDUCATION_3",
            "MARRIAGE_1",
            "MARRIAGE_2",
            "MARRIAGE_3",
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
            "default-payment-next-month",
        ]

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

        if split == "Sex":
            sens_attrs = ["SEX"]
            s_prefix = ["SEX"]
            class_labels = ["default-payment-next-month"]
            class_label_prefix = ["default-payment-next-month"]
        elif split == "Custom":
            pass
        else:
            raise NotImplementedError

    return Dataset(
        name=f"Credit {split}",
        num_samples=30000,
        filename_or_path="UCI_Credit_Card.csv",
        features=features,
        cont_features=continuous_features,
        s_prefix=s_prefix,
        sens_attrs=sens_attrs,
        class_label_prefix=class_label_prefix,
        class_labels=class_labels,
        discrete_only=discrete_only,
    )
