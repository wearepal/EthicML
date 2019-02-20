"""
Class to describe features of the Adult dataset
"""
from typing import Dict, List

from .dataset import Dataset
from .load import filter_features_by_prefixes


# Can't disable duplicate code warning on abstract methods, so disabling all for this file (for now)
# pylint: disable-all


class German(Dataset):
    features: List[str]
    y_prefix: List[str]
    y_labels: List[str]
    s_prefix: List[str]
    sens_attrs: List[str]
    _cont_features: List[str]
    _disc_features: List[str]

    def __init__(self, split: str = "Sex", discrete_only: bool = False):
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

        self._cont_features = [
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
            self.y_labels = ['credit']
            self.y_prefix = ['credit']
        else:
            raise NotImplementedError

        self.conc_features: List[str] = self.s_prefix + self.y_prefix
        self._disc_features = [item for item in filter_features_by_prefixes(self.features, self.conc_features)
                               if item not in self._cont_features]

    @property
    def name(self) -> str:
        return "German"

    @property
    def filename(self) -> str:
        return "german.csv"

    @property
    def feature_split(self) -> Dict[str, List[str]]:

        conc_features: List[str]
        if self.discrete_only:
            conc_features = self.s_prefix + self.y_prefix + self.continuous_features
        else:
            conc_features = self.s_prefix + self.y_prefix

        return {
            "x": filter_features_by_prefixes(self.features, conc_features),
            "s": self.sens_attrs,
            "y": self.y_labels
        }

    def set_s(self, sens_attrs: List[str]):
        self.sens_attrs = sens_attrs

    def set_s_prefix(self, sens_attr_prefixs: List[str]):
        self.s_prefix = sens_attr_prefixs

    def set_y(self, labels: List[str]):
        self.y_labels = labels

    def set_y_prefix(self, label_prefixs):
        self.y_prefix = label_prefixs

    @property
    def continuous_features(self) -> List[str]:
        return self._cont_features

    @property
    def discrete_features(self) -> List[str]:
        return self._disc_features
