"""Class to describe the 'LSAC Law School Admissions' dataset.

This dataset comes from Kusner et al Counterfactual Fairness.

Link to repo: https://github.com/mkusner/counterfactual-fairness/


@inproceedings{NIPS2017_a486cd07,
 author = {Kusner, Matt J and Loftus, Joshua and Russell, Chris and Silva, Ricardo},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Counterfactual Fairness},
 url = {https://proceedings.neurips.cc/paper/2017/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf},
 volume = {30},
 year = {2017}
}
"""
from enum import Enum
from typing import Mapping, Union
from typing_extensions import Literal

from ..dataset import Dataset
from ..util import LabelGroup, flatten_dict, simple_spec

__all__ = ["law"]


class LawSplits(Enum):
    SEX = "Sex"
    RACE = "Race"
    SEX_RACE = "Sex-Race"
    CUSTOM = "Custom"


VALID_STRS = Literal[tuple([e.value for e in LawSplits])]  # type: ignore[misc]


def law(
    split: Union[LawSplits, VALID_STRS] = "Sex",  # type: ignore[valid-type]
    discrete_only: bool = False,
) -> Dataset:
    """LSAC Law School dataset."""
    _split = LawSplits(split)
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

    if _split is LawSplits.SEX:
        sens_attr_spec: Union[str, Mapping[str, LabelGroup]] = "Sex_1"
        s_prefix = ["Sex", "Race"]
        class_label_spec = "PF_1"
        class_label_prefix = ["PF"]
    elif _split is LawSplits.RACE:
        sens_attr_spec = "Race_White"
        s_prefix = ["Race", "Sex"]
        class_label_spec = "PF_1"
        class_label_prefix = ["PF"]
    elif _split is LawSplits.SEX_RACE:
        sens_attr_spec = simple_spec({"Sex": ["Sex_1"], "Race": disc_feature_groups["Race"]})
        s_prefix = ["Race", "Sex"]
        class_label_spec = "PF_1"
        class_label_prefix = ["PF"]
    elif _split is LawSplits.CUSTOM:
        sens_attr_spec = ""
        s_prefix = []
        class_label_spec = ""
        class_label_prefix = []
    else:
        raise NotImplementedError

    name = f"Law {_split.value}"

    return Dataset(
        name=name,
        num_samples=21_791,
        features=discrete_features + continuous_features,
        cont_features=continuous_features,
        sens_attr_spec=sens_attr_spec,
        class_label_spec=class_label_spec,
        filename_or_path="law.csv.zip",
        s_prefix=s_prefix,
        class_label_prefix=class_label_prefix,
        discrete_only=discrete_only,
        discrete_feature_groups=disc_feature_groups,
    )
