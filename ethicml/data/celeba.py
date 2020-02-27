"""Class to describe attributes of the CelebA dataset."""
import warnings
from typing import Callable, Dict, List, Optional, cast

from typing_extensions import Literal

from ethicml.common import implements
from ethicml.preprocessing import SequentialSplit, get_biased_subset
from ethicml.vision import TorchCelebA

from .load import load_data
from .dataset import Dataset

__all__ = ["CELEBATTRS", "CelebA", "create_celeba_dataset"]


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


class CelebA(Dataset):
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
            "hair_type": ["Straight_Hair", "Wavy_Hair"],
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


def create_celeba_dataset(
    root: str,
    biased: bool,
    mixing_factor: float,
    unbiased_pcnt: float,
    sens_attr_name: CELEBATTRS,
    target_attr_name: CELEBATTRS,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    seed: int = 42,
) -> TorchCelebA:
    """Create a CelebA dataset object.

    Args:
        root: Root directory where images are downloaded to.
        biased: Wheher to artifically bias the dataset according to the mixing factor. See
                :func:`get_biased_subset()` for more details.
        mixing_factor: Mixing factor used to generate the biased subset of the data.
        sens_attr_name: Attribute(s) to set as the sensitive attribute. Biased sampling cannot be
                        performed if multiple sensitive attributes are specified.
        unbiased_pcnt: Percentage of the dataset to set aside as the 'unbiased' split.
        target_attr_name: Attribute to set as the target attribute.
        transform: A function/transform that  takes in an PIL image and returns a transformed
                   version. E.g, `transforms.ToTensor`
        target_transform: A function/transform that takes in the target and transforms it.
        download: If true, downloads the dataset from the internet and puts it in root
                  directory. If dataset is already downloaded, it is not downloaded again.
        seed: Random seed used to sample biased subset.
    """
    sens_attr_name = cast(CELEBATTRS, sens_attr_name.capitalize())
    target_attr_name = cast(CELEBATTRS, target_attr_name.capitalize())

    all_dt = load_data(CelebA(label=target_attr_name, sens_attr=sens_attr_name))

    if sens_attr_name == target_attr_name:
        warnings.warn("Same attribute specified for both the sensitive and target attribute.")

    # NOTE: the sequential split does not shuffle
    unbiased_dt, biased_dt, _ = SequentialSplit(train_percentage=unbiased_pcnt)(all_dt)

    if biased:
        biased_dt, _ = get_biased_subset(
            data=biased_dt, mixing_factor=mixing_factor, unbiased_pcnt=0, seed=seed
        )
        return TorchCelebA(
            data=biased_dt,
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
    else:
        return TorchCelebA(
            data=unbiased_dt,
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
