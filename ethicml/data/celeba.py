"""Class to describe attributes of the CelebA dataset."""
from typing import Dict, List

from typing_extensions import Literal

from ethicml.common import implements

from .dataset import Dataset

CELEBATTRS = Literal[
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
        self, label: CELEBATTRS = "Smiling", sens_attr: CELEBATTRS = "Male",
    ):
        """Init CelebA dataset."""
        disc_feature_groups = {
            "beard": ["5_o_Clock_Shadow", "Goatee", "Mustache", "No_Beard"],
            "hair_color": ["Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"],
            # not yet sorted into categories:
            "Arched_Eyebrows": ["Arched_Eyebrows"],
            "Attractive": ["Attractive"],
            "Bags_Under_Eyes": ["Bags_Under_Eyes"],
            "Bangs": ["Bangs"],
            "Big_Lips": ["Big_Lips"],
            "Big_Nose": ["Big_Nose"],
            "Blurry": ["Blurry"],
            "Bushy_Eyebrows": ["Bushy_Eyebrows"],
            "Chubby": ["Chubby"],
            "Double_Chin": ["Double_Chin"],
            "Eyeglasses": ["Eyeglasses"],
            "Heavy_Makeup": ["Heavy_Makeup"],
            "High_Cheekbones": ["High_Cheekbones"],
            "Male": ["Male"],
            "Mouth_Slightly_Open": ["Mouth_Slightly_Open"],
            "Narrow_Eyes": ["Narrow_Eyes"],
            "Oval_Face": ["Oval_Face"],
            "Pale_Skin": ["Pale_Skin"],
            "Pointy_Nose": ["Pointy_Nose"],
            "Receding_Hairline": ["Receding_Hairline"],
            "Rosy_Cheeks": ["Rosy_Cheeks"],
            "Sideburns": ["Sideburns"],
            "Smiling": ["Smiling"],
            "Straight_Hair": ["Straight_Hair"],
            "Wavy_Hair": ["Wavy_Hair"],
            "Wearing_Earrings": ["Wearing_Earrings"],
            "Wearing_Hat": ["Wearing_Hat"],
            "Wearing_Lipstick": ["Wearing_Lipstick"],
            "Wearing_Necklace": ["Wearing_Necklace"],
            "Wearing_Necktie": ["Wearing_Necktie"],
            "Young": ["Young"],
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
