"""Class to describe features of the German dataset."""
from ..dataset import Dataset

__all__ = ["german"]


def german(split: str = "Sex", discrete_only: bool = False) -> Dataset:
    """German credit dataset."""
    if True:  # pylint: disable=using-constant-test
        features = [
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

        continuous_features = [
            "month",
            "credit-amount",
            "investment-as-income-percentage",
            "residence-since",
            "age",
            "number-of-credits",
            "people-liable-for",
        ]

        if split == "Sex":
            sens_attr_spec = "sex"
            s_prefix = ["sex"]
            class_label_spec = "credit-label"
            class_label_prefix = ["credit-label"]
        else:
            raise NotImplementedError

    return Dataset(
        name=f"German {split}",
        num_samples=1000,
        filename_or_path="german.csv",
        features=features,
        cont_features=continuous_features,
        s_prefix=s_prefix,
        sens_attr_spec=sens_attr_spec,
        class_label_prefix=class_label_prefix,
        class_label_spec=class_label_spec,
        discrete_only=discrete_only,
    )
