"""Generated faces."""
from pathlib import Path
from typing import Optional, Tuple

from typing_extensions import Literal

from ..dataset import Dataset
from ..util import flatten_dict, simple_spec

__all__ = ["GenfacesAttributes", "genfaces"]


GenfacesAttributes = Literal[
    "gender", "age", "ethnicity", "eye_color", "hair_color", "hair_length", "emotion"
]

_BASE_FOLDER = "genfaces"
_SUBDIR = "images"
_FILE_ID = "14WTxAEe5-UyXqxJlnAOV3rVZz-ebJO1g"
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
    discrete_features = flatten_dict(disc_feature_groups)
    assert sens_attr in disc_feature_groups
    assert label in disc_feature_groups
    continuous_features = ["filename"]

    base = root / _BASE_FOLDER
    img_dir = base / _SUBDIR
    if download:
        _download(base)
    elif check_integrity:
        if not _check_integrity(base):
            return None, img_dir

    dataset_obj = Dataset(
        name=f"GenFaces, s={sens_attr}, y={label}",
        sens_attr_spec=simple_spec({sens_attr: disc_feature_groups[sens_attr]}),
        s_prefix=[sens_attr],
        class_label_spec=simple_spec({label: disc_feature_groups[label]}),
        class_label_prefix=[label],
        discrete_feature_groups=disc_feature_groups,
        features=discrete_features + continuous_features,
        cont_features=continuous_features,
        num_samples=148_285,
        filename_or_path="genfaces.csv.zip",
        discrete_only=False,
        discard_non_one_hot=True,
    )
    return dataset_obj, img_dir


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
