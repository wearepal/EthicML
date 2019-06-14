"""
Class to describe features of the German dataset
"""

from .dataset import Dataset


class German(Dataset):
    """German credit dataset"""

    def __init__(self, split: str = "Sex", discrete_only: bool = False):
        super().__init__()
        self.discrete_only = discrete_only
        self.features = [
            "month",
            "credit-amount",
            "investment-as-income-percentage",
            "sex",
            "residence-since",
            "age",
            "number-of-credits",
            "people-liable-for",
            "credit-label",
            "status_A11",
            "status_A12",
            "status_A13",
            "status_A14",
            "credit-history_A30",
            "credit-history_A31",
            "credit-history_A32",
            "credit-history_A33",
            "credit-history_A34",
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
            "savings_A61",
            "savings_A62",
            "savings_A63",
            "savings_A64",
            "savings_A65",
            "employment_A71",
            "employment_A72",
            "employment_A73",
            "employment_A74",
            "employment_A75",
            "other-debtors_A101",
            "other-debtors_A102",
            "other-debtors_A103",
            "property_A121",
            "property_A122",
            "property_A123",
            "property_A124",
            "installment-plans_A141",
            "installment-plans_A142",
            "installment-plans_A143",
            "housing_A151",
            "housing_A152",
            "housing_A153",
            "skill-level_A171",
            "skill-level_A172",
            "skill-level_A173",
            "skill-level_A174",
            "telephone_A191",
            "telephone_A192",
            "foreign-worker_A201",
            "foreign-worker_A202",
        ]

        self.continuous_features = [
            "month",
            "credit-amount",
            "investment-as-income-percentage",
            "residence-since",
            "age",
            "number-of-credits",
            "people-liable-for",
        ]

        if split == "Sex":
            self.sens_attrs = ["sex"]
            self.s_prefix = ["sex"]
            self.class_labels = ["credit-label"]
            self.class_label_prefix = ["credit-label"]
        else:
            raise NotImplementedError

    @property
    def name(self) -> str:
        return "German"

    @property
    def filename(self) -> str:
        return "german.csv"
