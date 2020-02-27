"""Generated faces."""
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, cast

from typing_extensions import Literal

from ethicml.common import implements
from ethicml.preprocessing import SequentialSplit, get_biased_subset
from ethicml.vision import TorchImageDataset

from .load import load_data
from .image_dataset import ImageDataset

__all__ = ["GenfacesAttributes", "GenFaces", "create_genfaces_dataset"]


GenfacesAttributes = Literal[
    "gender", "age", "ethnicity", "eye_color", "hair_color", "hair_length", "emotion",
]

_BASE_FOLDER = "genfaces"
_SUBDIR = "images"
_FILE_ID = "1rfwiDmsw37IDnMSWKx5gTc_CZ_Dh5d5g"
_ZIP_FILE = "genfaces_info.zip"


class GenFaces(ImageDataset):
    """Generated Faces dataset."""

    _disc_feature_groups: Dict[str, List[str]]

    def __init__(
        self,
        download_dir: str,
        label: GenfacesAttributes = "emotion",
        sens_attr: GenfacesAttributes = "gender",
    ):
        """Init GenFaces dataset."""
        self.root = Path(download_dir)
        disc_feature_groups = {
            "gender": ["gender_male"],
            "age": ["age_young-adult"],
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
            "emotion": ["emotion_joy"],
        }
        super().__init__(
            name=f"GenFaces, s={sens_attr}, y={label}",
            disc_feature_groups=disc_feature_groups,
            continuous_features=["filename"],
            length=148_285,
            csv_file="genfaces.csv.zip",
            img_dir=self.root / _BASE_FOLDER / _SUBDIR,
        )
        assert sens_attr in disc_feature_groups
        assert label in disc_feature_groups

        self.sens_attrs = disc_feature_groups[sens_attr]
        self.s_prefix = [sens_attr]
        self.class_labels = disc_feature_groups[label]
        self.class_label_prefix = [label]

    @implements(ImageDataset)
    def check_integrity(self) -> bool:
        base = self.root / _BASE_FOLDER
        return (base / _SUBDIR).is_dir()

    @implements(ImageDataset)
    def download(self) -> None:
        """Attempt to download data if files cannot be found in the base folder."""
        import zipfile
        from torchvision.datasets.utils import download_file_from_google_drive

        base = self.root / _BASE_FOLDER

        if self.check_integrity():
            print("Files already downloaded and verified")
            return

        fpath = base / _ZIP_FILE
        download_file_from_google_drive(fpath)

        with zipfile.ZipFile(fpath, "r") as fhandle:
            fhandle.extractall(str(base))


def create_genfaces_dataset(
    root: str,
    biased: bool,
    mixing_factor: float,
    unbiased_pcnt: float,
    sens_attr_name: GenfacesAttributes,
    target_attr_name: GenfacesAttributes,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    seed: int = 42,
) -> TorchImageDataset:
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
    sens_attr_name = cast(GenfacesAttributes, sens_attr_name.lower())
    target_attr_name = cast(GenfacesAttributes, target_attr_name.lower())

    gen_faces = GenFaces(download_dir=root, label=target_attr_name, sens_attr=sens_attr_name)
    if download:
        gen_faces.download()
    else:
        gen_faces.check_integrity()
    base_dir = gen_faces.img_dir
    all_dt = load_data(gen_faces)

    if sens_attr_name == target_attr_name:
        warnings.warn("Same attribute specified for both the sensitive and target attribute.")

    # NOTE: the sequential split does not shuffle
    unbiased_dt, biased_dt, _ = SequentialSplit(train_percentage=unbiased_pcnt)(all_dt)

    if biased:
        biased_dt, _ = get_biased_subset(
            data=biased_dt, mixing_factor=mixing_factor, unbiased_pcnt=0, seed=seed
        )
        return TorchImageDataset(
            data=biased_dt, root=base_dir, transform=transform, target_transform=target_transform,
        )
    else:
        return TorchImageDataset(
            data=unbiased_dt, root=base_dir, transform=transform, target_transform=target_transform,
        )
