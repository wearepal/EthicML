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
            'month',
            'credit_amount',
            'investment_as_income_percentage',
            'sex',
            'residence_since',
            'age',
            'number_of_credits',
            'people_liable_for',
            'credit',
            'status_A11',
            'status_A12',
            'status_A13',
            'status_A14',
            'credit_history_A30',
            'credit_history_A31',
            'credit_history_A32',
            'credit_history_A33',
            'credit_history_A34',
            'purpose_A40',
            'purpose_A41',
            'purpose_A410',
            'purpose_A42',
            'purpose_A43',
            'purpose_A44',
            'purpose_A45',
            'purpose_A46',
            'purpose_A48',
            'purpose_A49',
            'savings_A61',
            'savings_A62',
            'savings_A63',
            'savings_A64',
            'savings_A65',
            'employment_A71',
            'employment_A72',
            'employment_A73',
            'employment_A74',
            'employment_A75',
            'other_debtors_A101',
            'other_debtors_A102',
            'other_debtors_A103',
            'property_A121',
            'property_A122',
            'property_A123',
            'property_A124',
            'installment_plans_A141',
            'installment_plans_A142',
            'installment_plans_A143',
            'housing_A151',
            'housing_A152',
            'housing_A153',
            'skill_level_A171',
            'skill_level_A172',
            'skill_level_A173',
            'skill_level_A174',
            'telephone_A191',
            'telephone_A192',
            'foreign_worker_A201',
            'foreign_worker_A202'
        ]

        self.continuous_features = [
            'month',
            'credit_amount',
            'investment_as_income_percentage',
            'residence_since',
            'age',
            'number_of_credits',
            'people_liable_for'
        ]

        if split == "Sex":
            self.sens_attrs = ['sex']
            self.s_prefix = ['sex']
            self.class_labels = ['credit']
            self.class_label_prefix = ['credit']
        else:
            raise NotImplementedError

    @property
    def name(self) -> str:
        return "German"

    @property
    def filename(self) -> str:
        return "german.csv"
