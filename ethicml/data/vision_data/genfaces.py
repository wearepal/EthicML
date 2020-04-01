"""Generated faces."""
import warnings
from pathlib import Path
from typing import Callable, List, Optional, cast, Tuple

from typing_extensions import Literal

from ethicml.preprocessing import ProportionalSplit, get_biased_subset
from ethicml.vision import TorchImageDataset

from ..dataset import Dataset

__all__ = ["GenfacesAttributes", "genfaces", "create_genfaces_dataset"]


GenfacesAttributes = Literal[
    "gender", "age", "ethnicity", "eye_color", "hair_color", "hair_length", "emotion",
]

_BASE_FOLDER = "genfaces"
_SUBDIR = "images"
_FILE_ID = "1TYeoJS-2Q2uU3BD9Ckf9aJOHbxLz5wFr"
_ZIP_FILE = "genfaces_info.zip"


def genfaces(
    download_dir: str,
    label: GenfacesAttributes = "emotion",
    sens_attr: GenfacesAttributes = "gender",
    download: bool = False,
    check_integrity: bool = True,
) -> Tuple[Optional[Dataset], Path]:
    """Generated Faces dataset."""
    root = Path(download_dir)
    disc_feature_groups = {
        "gender": ["gender_male"],
        "age": ["age_young-adult"],
        "ethnicity": ["ethnicity_asian", "ethnicity_black", "ethnicity_latino", "ethnicity_white"],
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
    discrete_features: List[str] = []
    for group in disc_feature_groups.values():
        discrete_features += group
    assert sens_attr in disc_feature_groups
    assert label in disc_feature_groups
    continuous_features = ["filename"]

    img_dir = root / _BASE_FOLDER / _SUBDIR
    if download:
        _download(root)
    elif check_integrity:
        if not _check_integrity(root):
            return None, img_dir

    return (
        Dataset(
            name=f"GenFaces, s={sens_attr}, y={label}",
            sens_attrs=disc_feature_groups[sens_attr],
            s_prefix=[sens_attr],
            class_labels=disc_feature_groups[label],
            class_label_prefix=[label],
            disc_feature_groups=disc_feature_groups,
            features=discrete_features + continuous_features,
            cont_features=continuous_features,
            num_samples=148_285,
            filename_or_path="genfaces.csv.zip",
            discrete_only=False,
        ),
        img_dir,
    )


def _check_integrity(base: Path) -> bool:
    return (base / _SUBDIR).is_dir()


def _download(base: Path) -> None:
    """Attempt to download data if files cannot be found in the base folder."""
    import zipfile
    from torchvision.datasets.utils import download_file_from_google_drive

    if _check_integrity(base):
        print("Files already downloaded and verified")
        return

    download_file_from_google_drive(_FILE_ID, str(base), _ZIP_FILE)

    fpath = base / _ZIP_FILE
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
    check_integrity: bool = True,
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
        check_integrity: If True, check whether the data has been downloaded correctly.
    """
    sens_attr_name = cast(GenfacesAttributes, sens_attr_name.lower())
    target_attr_name = cast(GenfacesAttributes, target_attr_name.lower())

    dataset, base_dir = genfaces(
        download_dir=root,
        label=target_attr_name,
        sens_attr=sens_attr_name,
        download=download,
        check_integrity=check_integrity,
    )
    assert dataset is not None
    all_dt = dataset.load()

    if sens_attr_name == target_attr_name:
        warnings.warn("Same attribute specified for both the sensitive and target attribute.")

    unbiased_dt, biased_dt, _ = ProportionalSplit(unbiased_pcnt, start_seed=seed)(all_dt)

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
