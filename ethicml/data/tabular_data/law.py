"""Class to describe the 'LSAC Law School Admissions' dataset.

This dataset comes from Kusner et al Counterfactual Fairness.

Link to repo: https://github.com/mkusner/counterfactual-fairness/

.. code-block:: bibtex

   @inproceedings{NIPS2017_a486cd07,
     author = {Kusner, Matt J and Loftus, Joshua and Russell, Chris and Silva, Ricardo},
     booktitle = {Advances in Neural Information Processing Systems},
     editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus
               and S. Vishwanathan and R. Garnett},
     pages = {},
     publisher = {Curran Associates, Inc.},
     title = {Counterfactual Fairness},
     url =
       {https://proceedings.neurips.cc/paper/2017/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf},
     volume = {30},
     year = {2017}
   }
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from ..dataset import LegacyDataset
from ..util import LabelGroup, flatten_dict, spec_from_binary_cols

__all__ = ["Law", "LawSplits"]


class LawSplits(Enum):
    """Splits for the Law dataset."""

    SEX = "Sex"
    RACE = "Race"
    SEX_RACE = "Sex-Race"
    CUSTOM = "Custom"


class _Law:
    Splits: TypeAlias = LawSplits
    """Shorthand for the Enum that defines the splits associated with this class."""


@dataclass
class Law(_Law, LegacyDataset):
    """LSAC Law School dataset."""

    split: LawSplits = _Law.Splits.SEX

    def __post_init__(self) -> None:
        disc_feature_groups = {
            "Sex": ["Sex_1", "Sex_2"],
            "Race": [
                "Race_Amerindian",
                "Race_Asian",
                "Race_Black",
                "Race_Hispanic",
                "Race_Mexican",
                "Race_Other",
                "Race_Puertorican",
                "Race_White",
            ],
            "PF": ["PF_0", "PF_1"],
        }
        discrete_features = flatten_dict(disc_feature_groups)

        continuous_features = ["LSAT", "UGPA", "ZFYA"]

        match self.split:
            case LawSplits.SEX:
                sens_attr_spec: str | Mapping[str, LabelGroup] = "Sex_1"
                s_prefix = ["Sex", "Race"]
                class_label_spec = "PF_1"
                class_label_prefix = ["PF"]
            case LawSplits.RACE:
                sens_attr_spec = "Race_White"
                s_prefix = ["Race", "Sex"]
                class_label_spec = "PF_1"
                class_label_prefix = ["PF"]
            case LawSplits.SEX_RACE:
                sens_attr_spec = spec_from_binary_cols(
                    {"Sex": ["Sex_1"], "Race": disc_feature_groups["Race"]}
                )
                s_prefix = ["Race", "Sex"]
                class_label_spec = "PF_1"
                class_label_prefix = ["PF"]
            case LawSplits.CUSTOM:
                sens_attr_spec = ""
                s_prefix = []
                class_label_spec = ""
                class_label_prefix = []
            case _:
                raise NotImplementedError

        name = f"Law {self.split.value}"

        super().__init__(
            name=name,
            num_samples=21_791,
            features=discrete_features + continuous_features,
            cont_features=continuous_features,
            sens_attr_spec=sens_attr_spec,
            class_label_spec=class_label_spec,
            filename_or_path="law.csv.zip",
            s_feature_groups=s_prefix,
            class_feature_groups=class_label_prefix,
            discrete_feature_groups=disc_feature_groups,
        )
