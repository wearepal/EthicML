"""Class to describe the 'UFRGS Entrance Exam and GPA Data'.

Persistent link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O35FW8

Data Description: Entrance exam scores of students applying to a university in Brazil
(Federal University of Rio Grande do Sul), along with the students' GPAs during the first three
semesters at university. In this dataset, each row contains anonymized information about an
applicant's scores on nine exams taken as part of the application process to the university, as
well as their corresponding GPA during the first three semesters at university.
The dataset has 43,303 rows, each corresponding to one student.
The columns correspond to:
1) Gender. 0 denotes female and 1 denotes male.
2) Score on physics exam
3) Score on biology exam
4) Score on history exam
5) Score on second language exam
6) Score on geography exam
7) Score on literature exam
8) Score on Portuguese essay exam
9) Score on math exam
10) Score on chemistry exam
11) Mean GPA during first three semesters at university, on a 4.0 scale.

We replace the mean GPA with a binary label Y representing whether the student’s GPA was above 3.0


```bibtex
@data{DVN/O35FW8_2019,
  author    = {Castro~da~Silva, Bruno},
  publisher = {Harvard Dataverse},
  title     = {{UFRGS Entrance Exam and GPA Data}},
  UNF       = {UNF:6:MQqEQGXiIfQTbS7q9QJ5uw==},
  year      = {2019},
  version   = {V1},
  doi       = {10.7910/DVN/O35FW8},
  url       = {https://doi.org/10.7910/DVN/O35FW8}
}
```
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Type

from ..dataset import LegacyDataset
from ..util import flatten_dict

__all__ = ["AdmissionsSplits", "Admissions"]


class AdmissionsSplits(Enum):
    """Splits for the Admissions dataset."""

    GENDER = "Gender"
    CUSTOM = "Custom"


@dataclass
class Admissions(LegacyDataset):
    """UFRGS Admissions dataset."""

    split: AdmissionsSplits = AdmissionsSplits.GENDER

    Splits: ClassVar[Type[AdmissionsSplits]] = AdmissionsSplits
    """Shorthand for the Enum that defines the splits associated with this class."""

    def __post_init__(self) -> None:
        disc_feature_groups = {"gender": ["gender"], "gpa": ["gpa"]}
        discrete_features = flatten_dict(disc_feature_groups)

        continuous_features = [
            "physics",
            "biology",
            "history",
            "language",
            "geography",
            "literature",
            "essay",
            "math",
            "chemistry",
        ]

        if self.split is AdmissionsSplits.GENDER:
            sens_attr_spec = "gender"
            s_prefix = ["gender"]
            class_label_spec = "gpa"
            class_label_prefix = ["gpa"]
        elif self.split is AdmissionsSplits.CUSTOM:
            sens_attr_spec = ""
            s_prefix = []
            class_label_spec = ""
            class_label_prefix = []
        else:
            raise NotImplementedError

        name = f"Admissions {self.split.value}"

        super().__init__(
            name=name,
            num_samples=43_303,
            features=discrete_features + continuous_features,
            cont_features=continuous_features,
            sens_attr_spec=sens_attr_spec,
            class_label_spec=class_label_spec,
            filename_or_path="admissions.csv.zip",
            s_feature_groups=s_prefix,
            class_feature_groups=class_label_prefix,
            discrete_feature_groups=disc_feature_groups,
        )
