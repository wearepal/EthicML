"""Class to describe attributes of the CelebA dataset."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from typing_extensions import Final, Literal

from ethicml import common

from ..dataset import Dataset
from ..util import LabelGroup, flatten_dict, label_spec_to_feature_list

__all__ = ["CelebAttrs", "celeba"]


CelebAttrs = Literal[
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

_BASE_FOLDER: Final = "celeba"

_FILE_LIST: Final = [
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


def celeba(
    download_dir: str,
    label: CelebAttrs = "Smiling",
    sens_attr: Union[CelebAttrs, Dict[str, LabelGroup]] = "Male",
    download: bool = False,
    check_integrity: bool = True,
) -> Tuple[Optional[Dataset], Path]:
    """Get CelebA dataset."""
    root = Path(download_dir)
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
    discrete_features = flatten_dict(disc_feature_groups)
    s_prefix: List[str]
    if isinstance(sens_attr, dict):
        s_prefix = label_spec_to_feature_list(sens_attr)
        assert all(feat in discrete_features for feat in s_prefix)
        name = "[" + ", ".join(sens_attr) + "]"
    else:
        assert sens_attr in discrete_features
        s_prefix = [sens_attr]
        name = sens_attr
    assert label in discrete_features
    continuous_features = ["filename"]

    base = root / _BASE_FOLDER
    img_dir = base / "img_align_celeba"
    if download:
        _download(base)
    elif check_integrity:
        if not _check_integrity(base):
            return None, img_dir
    dataset_obj = Dataset(
        name=f"CelebA, s={name}, y={label}",
        sens_attr_spec=sens_attr,
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

    for (_, md5, filename) in _FILE_LIST:
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

    for (file_id, md5, filename) in _FILE_LIST:
        download_file_from_google_drive(file_id, str(base), filename, md5)

    with zipfile.ZipFile(base / "img_align_celeba.zip", "r") as fhandle:
        fhandle.extractall(str(base))
    return
