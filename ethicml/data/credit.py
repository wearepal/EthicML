"""
Class to describe features of the Adult dataset
"""

from .dataset import Dataset


class Credit(Dataset):
    """
    UCL Credit Card dataset
    """

    def __init__(self, split: str = "Sex", discrete_only: bool = False):
        super().__init__()
        self.discrete_only = discrete_only
        self.features = [
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

        self.continuous_features = [
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
            self.sens_attrs = ["SEX"]
            self.s_prefix = ["SEX"]
            self.class_labels = ["default-payment-next-month"]
            self.class_label_prefix = ["default-payment-next-month"]
        elif split == "Custom":
            pass
        else:
            raise NotImplementedError

    @property
    def name(self) -> str:
        return "Credit"

    @property
    def filename(self) -> str:
        return "UCI_Credit_Card.csv"
