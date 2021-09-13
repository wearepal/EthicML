"""Class to describe attributes of the CelebA dataset."""
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Final

from ethicml import common

from ..dataset import Dataset
from ..util import LabelGroup, flatten_dict, label_spec_to_feature_list

__all__ = ["CELEBA_BASE_FOLDER", "CELEBA_FILE_LIST", "CelebAttr", "celeba"]


class CelebAttr(Enum):
    """Enum for CelebA attributes."""

    Five_o_Clock_Shadow = auto()
    Arched_Eyebrows = auto()
    Attractive = auto()
    Bags_Under_Eyes = auto()
    Bald = auto()
    Bangs = auto()
    Big_Lips = auto()
    Big_Nose = auto()
    Black_Hair = auto()
    Blond_Hair = auto()
    Blurry = auto()
    Brown_Hair = auto()
    Bushy_Eyebrows = auto()
    Chubby = auto()
    Double_Chin = auto()
    Eyeglasses = auto()
    Goatee = auto()
    Gray_Hair = auto()
    Heavy_Makeup = auto()
    High_Cheekbones = auto()
    Male = auto()
    Mouth_Slightly_Open = auto()
    Mustache = auto()
    Narrow_Eyes = auto()
    No_Beard = auto()
    Oval_Face = auto()
    Pale_Skin = auto()
    Pointy_Nose = auto()
    Receding_Hairline = auto()
    Rosy_Cheeks = auto()
    Sideburns = auto()
    Smiling = auto()
    Straight_Hair = auto()
    Wavy_Hair = auto()
    Wearing_Earrings = auto()
    Wearing_Hat = auto()
    Wearing_Lipstick = auto()
    Wearing_Necklace = auto()
    Wearing_Necktie = auto()
    Young = auto()


CELEBA_BASE_FOLDER: Final = "celeba"
"""The data is downloaded to `download_dir` / `CELEBA_BASE_FOLDER`."""

CELEBA_FILE_LIST: Final = [
    (
        "1zmsC4yvw-e089uHXj5EdP0BSZ0AlDQRR",  # File ID
        "00d2c5bc6d35e252742224ab0c1e8fcb",  # MD5 Hash
        "img_align_celeba.zip",  # Filename
    ),
    (
        "1gxmFoeEPgF9sT65Wpo85AnHl3zsQ4NvS",
        "75e246fa4810816ffd6ee81facbd244c",
        "list_attr_celeba.txt",
    ),
    (
        "1ih_VMokoI774ErNWrb26lDeWlanUBpnX",
        "d32c9cbf5e040fd4025c592c306e6668",
        "list_eval_partition.txt",
    ),
]
"""Google drive IDs, MD5 hashes and filenames for the CelebA files."""


def celeba(
    download_dir: str,
    label: CelebAttr = CelebAttr.Smiling,
    sens_attr: Union[CelebAttr, Dict[str, LabelGroup]] = CelebAttr.Male,
    download: bool = False,
    check_integrity: bool = True,
) -> Tuple[Optional[Dataset], Path]:
    """Get CelebA dataset."""
    root = Path(download_dir)
    disc_feature_groups = {
        "beard": [
            CelebAttr.Five_o_Clock_Shadow,
            CelebAttr.Goatee,
            CelebAttr.Mustache,
            CelebAttr.No_Beard,
        ],
        "hair_color": [
            CelebAttr.Bald,
            CelebAttr.Black_Hair,
            CelebAttr.Blond_Hair,
            CelebAttr.Brown_Hair,
            CelebAttr.Gray_Hair,
        ],
        # not yet sorted into categories:
        "Arched_Eyebrows": [CelebAttr.Arched_Eyebrows],
        "Attractive": [CelebAttr.Attractive],
        "Bags_Under_Eyes": [CelebAttr.Bags_Under_Eyes],
        "Bangs": [CelebAttr.Bangs],
        "Big_Lips": [CelebAttr.Big_Lips],
        "Big_Nose": [CelebAttr.Big_Nose],
        "Blurry": [CelebAttr.Blurry],
        "Bushy_Eyebrows": [CelebAttr.Bushy_Eyebrows],
        "Chubby": [CelebAttr.Chubby],
        "Double_Chin": [CelebAttr.Double_Chin],
        "Eyeglasses": [CelebAttr.Eyeglasses],
        "Heavy_Makeup": [CelebAttr.Heavy_Makeup],
        "High_Cheekbones": [CelebAttr.High_Cheekbones],
        "Male": [CelebAttr.Male],
        "Mouth_Slightly_Open": [CelebAttr.Mouth_Slightly_Open],
        "Narrow_Eyes": [CelebAttr.Narrow_Eyes],
        "Oval_Face": [CelebAttr.Oval_Face],
        "Pale_Skin": [CelebAttr.Pale_Skin],
        "Pointy_Nose": [CelebAttr.Pointy_Nose],
        "Receding_Hairline": [CelebAttr.Receding_Hairline],
        "Rosy_Cheeks": [CelebAttr.Rosy_Cheeks],
        "Sideburns": [CelebAttr.Sideburns],
        "Smiling": [CelebAttr.Smiling],
        "hair_type": [CelebAttr.Straight_Hair, CelebAttr.Wavy_Hair],
        "Wearing_Earrings": [CelebAttr.Wearing_Earrings],
        "Wearing_Hat": [CelebAttr.Wearing_Earrings],
        "Wearing_Lipstick": [CelebAttr.Wearing_Lipstick],
        "Wearing_Necklace": [CelebAttr.Wearing_Necklace],
        "Wearing_Necktie": [CelebAttr.Wearing_Necktie],
        "Young": [CelebAttr.Young],
    }
    discrete_features = flatten_dict(disc_feature_groups)
    s_prefix: List[str]
    if isinstance(sens_attr, dict):
        s_prefix = label_spec_to_feature_list(sens_attr)
        assert all(feat in discrete_features for feat in s_prefix)
        name = "[" + ", ".join(sens_attr) + "]"
    else:
        assert sens_attr in discrete_features
        s_prefix = [sens_attr.name]
        name = sens_attr
    assert label in discrete_features
    continuous_features = ["filename"]

    base = root / CELEBA_BASE_FOLDER
    img_dir = base / "img_align_celeba"
    if download:
        _download(base)
    elif check_integrity and not _check_integrity(base):
        return None, img_dir
    dataset_obj = Dataset(
        name=f"CelebA, s={name}, y={label}",
        sens_attr_spec=sens_attr.name if isinstance(sens_attr, CelebAttr) else sens_attr,
        s_prefix=s_prefix,
        class_label_spec=label,
        class_label_prefix=[label],
        discrete_feature_groups=disc_feature_groups,
        features=discrete_features + continuous_features,
        cont_features=continuous_features,
        num_samples=202599,
        filename_or_path="celeba.csv.zip",
        discrete_only=False,
        discard_non_one_hot=True,
        map_to_binary=True,
    )
    return dataset_obj, img_dir


def _check_integrity(base: Path) -> bool:
    if not common.TORCHVISION_AVAILABLE:
        raise RuntimeError("Need torchvision to download data.")
    from torchvision.datasets.utils import check_integrity

    for (_, md5, filename) in CELEBA_FILE_LIST:
        fpath = base / filename
        ext = fpath.suffix
        # Allow original archive to be deleted (zip and 7z)
        # Only need the extracted images
        if ext not in [".zip", ".7z"] and not check_integrity(str(fpath), md5):
            return False

    # Should check a hash of the images
    return (base / "img_align_celeba").is_dir()


def _download(base: Path) -> None:
    """Attempt to download data if files cannot be found in the base folder."""
    if not common.TORCHVISION_AVAILABLE:
        raise RuntimeError("Need torchvision to download data.")
    import zipfile

    from torchvision.datasets.utils import download_file_from_google_drive

    if _check_integrity(base):
        print("Files already downloaded and verified")
        return

    for (file_id, md5, filename) in CELEBA_FILE_LIST:
        download_file_from_google_drive(file_id, str(base), filename, md5)

    with zipfile.ZipFile(base / "img_align_celeba.zip", "r") as fhandle:
        fhandle.extractall(str(base))
