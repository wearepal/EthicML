"""
Class to describe features of the Adult dataset
"""

from .dataset import Dataset


class Adult(Dataset):
    """
    UCL Adult dataset
    """

    def __init__(self, split: str = "Sex", discrete_only: bool = False):
        super().__init__()
        self.discrete_only = discrete_only
        self.features = [
            "age",
            "capital-gain",
            "capital-loss",
            "education-num",
            "education_10th",
            "education_11th",
            "education_12th",
            "education_1st-4th",
            "education_5th-6th",
            "education_7th-8th",
            "education_9th",
            "education_Assoc-acdm",
            "education_Assoc-voc",
            "education_Bachelors",
            "education_Doctorate",
            "education_HS-grad",
            "education_Masters",
            "education_Preschool",
            "education_Prof-school",
            "education_Some-college",
            "hours-per-week",
            "marital-status_Divorced",
            "marital-status_Married-AF-spouse",
            "marital-status_Married-civ-spouse",
            "marital-status_Married-spouse-absent",
            "marital-status_Never-married",
            "marital-status_Separated",
            "marital-status_Widowed",
            "native-country_Cambodia",
            "native-country_Canada",
            "native-country_China",
            "native-country_Columbia",
            "native-country_Cuba",
            "native-country_Dominican-Republic",
            "native-country_Ecuador",
            "native-country_El-Salvador",
            "native-country_England",
            "native-country_France",
            "native-country_Germany",
            "native-country_Greece",
            "native-country_Guatemala",
            "native-country_Haiti",
            "native-country_Holand-Netherlands",
            "native-country_Honduras",
            "native-country_Hong",
            "native-country_Hungary",
            "native-country_India",
            "native-country_Iran",
            "native-country_Ireland",
            "native-country_Italy",
            "native-country_Jamaica",
            "native-country_Japan",
            "native-country_Laos",
            "native-country_Mexico",
            "native-country_Nicaragua",
            "native-country_Outlying-US(Guam-USVI-etc)",
            "native-country_Peru",
            "native-country_Philippines",
            "native-country_Poland",
            "native-country_Portugal",
            "native-country_Puerto-Rico",
            "native-country_Scotland",
            "native-country_South",
            "native-country_Taiwan",
            "native-country_Thailand",
            "native-country_Trinadad&Tobago",
            "native-country_United-States",
            "native-country_Vietnam",
            "native-country_Yugoslavia",
            "occupation_Adm-clerical",
            "occupation_Armed-Forces",
            "occupation_Craft-repair",
            "occupation_Exec-managerial",
            "occupation_Farming-fishing",
            "occupation_Handlers-cleaners",
            "occupation_Machine-op-inspct",
            "occupation_Other-service",
            "occupation_Priv-house-serv",
            "occupation_Prof-specialty",
            "occupation_Protective-serv",
            "occupation_Sales",
            "occupation_Tech-support",
            "occupation_Transport-moving",
            "race_Amer-Indian-Eskimo",
            "race_Asian-Pac-Islander",
            "race_Black",
            "race_Other",
            "race_White",
            "relationship_Husband",
            "relationship_Not-in-family",
            "relationship_Other-relative",
            "relationship_Own-child",
            "relationship_Unmarried",
            "relationship_Wife",
            "salary_<=50K",
            "salary_>50K",
            "sex_Female",
            "sex_Male",
            "workclass_Federal-gov",
            "workclass_Local-gov",
            "workclass_Private",
            "workclass_Self-emp-inc",
            "workclass_Self-emp-not-inc",
            "workclass_State-gov",
            "workclass_Without-pay",
        ]

        self.continuous_features = [
            "age",
            "capital-gain",
            "capital-loss",
            "education-num",
            "hours-per-week",
        ]

        if split == "Sex":
            self.sens_attrs = ["sex_Male"]
            self.s_prefix = ["sex"]
            self.class_labels = ["salary_>50K"]
            self.class_label_prefix = ["salary"]
        elif split == "Race":
            self.sens_attrs = [
                "race_Amer-Indian-Eskimo",
                "race_Asian-Pac-Islander",
                "race_Black",
                "race_Other",
                "race_White",
            ]
            self.s_prefix = ["race"]
            self.class_labels = ["salary_>50K"]
            self.class_label_prefix = ["salary"]
        elif split == "Custom":
            pass
        elif split == "Race-Sex":
            self.sens_attrs = [
                "sex_Male",
                "race_Amer-Indian-Eskimo",
                "race_Asian-Pac-Islander",
                "race_Black",
                "race_Other",
                "race_White",
            ]
            self.s_prefix = ["race", "sex"]
            self.class_labels = ["salary_>50K"]
            self.class_label_prefix = ["salary"]
        elif split == "Nationality":
            self.sens_attrs = [
                "native-country_Cambodia",
                "native-country_Canada",
                "native-country_China",
                "native-country_Columbia",
                "native-country_Cuba",
                "native-country_Dominican-Republic",
                "native-country_Ecuador",
                "native-country_El-Salvador",
                "native-country_England",
                "native-country_France",
                "native-country_Germany",
                "native-country_Greece",
                "native-country_Guatemala",
                "native-country_Haiti",
                "native-country_Holand-Netherlands",
                "native-country_Honduras",
                "native-country_Hong",
                "native-country_Hungary",
                "native-country_India",
                "native-country_Iran",
                "native-country_Ireland",
                "native-country_Italy",
                "native-country_Jamaica",
                "native-country_Japan",
                "native-country_Laos",
                "native-country_Mexico",
                "native-country_Nicaragua",
                "native-country_Outlying-US(Guam-USVI-etc)",
                "native-country_Peru",
                "native-country_Philippines",
                "native-country_Poland",
                "native-country_Portugal",
                "native-country_Puerto-Rico",
                "native-country_Scotland",
                "native-country_South",
                "native-country_Taiwan",
                "native-country_Thailand",
                "native-country_Trinadad&Tobago",
                "native-country_United-States",
                "native-country_Vietnam",
                "native-country_Yugoslavia",
            ]
            self.s_prefix = ["native-country"]
            self.class_labels = ["salary_>50K"]
            self.class_label_prefix = ["salary"]
        else:
            raise NotImplementedError

    @property
    def name(self) -> str:
        return "Adult"

    @property
    def filename(self) -> str:
        return "adult.csv"
