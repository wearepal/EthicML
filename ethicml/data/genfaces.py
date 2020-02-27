import warnings
from typing import Callable, Dict, List, Optional, cast

from typing_extensions import Literal

from ethicml.common import implements
from ethicml.preprocessing import SequentialSplit, get_biased_subset
from ethicml.vision import TorchGenFaces

from . import Dataset, load_data

__all__ = ["GENFACES_ATTRIBUTES", "GenFaces"]


GENFACES_ATTRIBUTES = Literal[
    "gender_female",
    "gender_male",
    "age_infant",
    "age_child",
    "age_adult",
    "age_young-adult",
    "age_adult",
    "age_elderly",
    "ethnicity_asian",
    "ethnicity_black",
    "ethnicity_latino",
    "ethnicity_white",
    "eye_color_blue",
    "eye_color_green",
    "eye_color_brown",
    "eye_color_gray",
    "hair_color_black",
    "hair_color_blond",
    "hair_color_brown",
    "hair_color_gray",
    "hair_color_white",
    "hair_length_long",
    "hair_length_medium",
    "hair_length_short",
    "emotion_joy",
    "emotion_neutral",
    "emotion_surprise",
]


class GenFaces(Dataset):
    """Generated Faces dataset."""

    _disc_feature_groups: Dict[str, List[str]]

    def __init__(
        self, label: GENFACES_ATTRIBUTES = "emotion", sens_attr: GENFACES_ATTRIBUTES = "gender",
    ):
        """Init GenFaces dataset."""
        disc_feature_groups = {
            "gender": ["gender_female", "gender_male"],
            "age": [
                "age_infant",
                "age_child",
                "age_adult",
                "age_young-adult",
                "age_adult",
                "age_elderly",
            ],
            "ethnicity": [
                "ethnicity_asian",
                "ethnicity_black",
                "ethnicity_latino",
                "ethnicity_white",
            ],
            "eye_color": ["eye_color_blue", "eye_color_brown", "eye_color_gray", "eye_color_green"],
            "hair_color": [
                "hair_color_black",
                "hair_color_blond",
                "hair_color_brown",
                "hair_color_gray",
                "hair_color_red",
            ],
            "hair_length": ["hair_length_long", "hair_length_medium", "hair_length_short"],
            "emotion": ["emotion_joy", "emotion_neutral", "emotion_surprise"],
        }
        super().__init__(disc_feature_groups=disc_feature_groups)
        discrete_features: List[str] = []
        for group in disc_feature_groups.values():
            discrete_features += group

        assert sens_attr in discrete_features
        assert label in discrete_features
        discrete_features.remove(sens_attr)
        discrete_features.remove(label)
        self.continuous_features = ["id"]
        self.features = self.continuous_features + discrete_features

        self.sens_attrs = [sens_attr]
        self.s_prefix = []
        self.class_labels = [label]
        self.class_label_prefix = []
        self.__name = f"GenFaces, s={sens_attr}, y={label}"

    @property
    def name(self) -> str:
        """Getter for dataset name."""
        return self.__name

    @property
    def filename(self) -> str:
        """Getter for filename."""
        return "genfaces.csv.zip"

    @implements(Dataset)
    def __len__(self) -> int:
        return 148285


def create_genfaces_dataset(
    root: str,
    biased: bool,
    mixing_factor: float,
    unbiased_pcnt: float,
    sens_attr_name: GENFACES_ATTRIBUTES,
    target_attr_name: GENFACES_ATTRIBUTES,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    seed: int = 42,
) -> TorchGenFaces:
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
    sens_attr_name = cast(GENFACES_ATTRIBUTES, sens_attr_name.lower())
    target_attr_name = cast(GENFACES_ATTRIBUTES, target_attr_name.lower())

    all_dt = load_data(GenFaces(label=target_attr_name, sens_attr=sens_attr_name))

    if sens_attr_name == target_attr_name:
        warnings.warn("Same attribute specified for both the sensitive and target attribute.")

    # NOTE: the sequential split does not shuffle
    unbiased_dt, biased_dt, _ = SequentialSplit(train_percentage=unbiased_pcnt)(all_dt)

    if biased:
        biased_dt, _ = get_biased_subset(
            data=biased_dt, mixing_factor=mixing_factor, unbiased_pcnt=0, seed=seed
        )
        return TorchGenFaces(
            data=biased_dt,
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
    else:
        return TorchGenFaces(
            data=unbiased_dt,
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
