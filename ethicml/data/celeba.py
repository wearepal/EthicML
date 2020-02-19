"""Class to describe attributes of the CelebA dataset."""
from typing import Dict, List

from typing_extensions import Literal

from ethicml.common import implements

from .dataset import Dataset

_CELEBATTRS = Literal[
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


class Celeba(Dataset):
    """CelebA dataset."""

    _disc_feature_groups: Dict[str, List[str]]

    def __init__(
        self, label: _CELEBATTRS = "Smiling", sens_attr: _CELEBATTRS = "Male",
    ):
        """Init CelebA dataset."""
        disc_feature_groups = {
            "hair_color": ["Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair",],
            "other": [
                "5_o_Clock_Shadow",
                "Arched_Eyebrows",
                "Attractive",
                "Bags_Under_Eyes",
                "Bangs",
                "Big_Lips",
                "Big_Nose",
                "Blurry",
                "Bushy_Eyebrows",
                "Chubby",
                "Double_Chin",
                "Eyeglasses",
                "Goatee",
                "Heavy_Makeup",
                "High_Cheekbones",
                "Male",
                "Mouth_Slightly_Open",
                "Mustache",
                "Narrow_Eyes",
                "No_Beard",
                "Oval_Face",
                "Pale_Skin",
                "Pointy_Nose",
                "Receding_Hairline",
                "Rosy_Cheeks",
                "Sideburns",
                "Smiling",
                "Straight_Hair",
                "Wavy_Hair",
                "Wearing_Earrings",
                "Wearing_Hat",
                "Wearing_Lipstick",
                "Wearing_Necklace",
                "Wearing_Necktie",
                "Young",
            ],
        }
        super().__init__(disc_feature_groups=disc_feature_groups)
        discrete_features: List[str] = []
        for group in disc_feature_groups.values():
            discrete_features += group

        assert sens_attr in discrete_features
        assert label in discrete_features
        discrete_features.remove(sens_attr)
        discrete_features.remove(label)
        self.continuous_features = ["filename"]
        self.features = self.continuous_features + discrete_features

        self.sens_attrs = [sens_attr]
        self.s_prefix = []
        self.class_labels = [label]
        self.class_label_prefix = []
        self.__name = f"CelebA, s={sens_attr}, y={label}"

    @property
    def name(self) -> str:
        """Getter for dataset name."""
        return self.__name

    @property
    def filename(self) -> str:
        """Getter for filename."""
        return "celeba.csv.zip"

    @implements(Dataset)
    def __len__(self) -> int:
        return 202599
