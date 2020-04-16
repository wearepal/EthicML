"""Class to describe attributes of the CelebA dataset."""
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, cast

from typing_extensions import Final, Literal

from ethicml.preprocessing import ProportionalSplit, get_biased_subset
from ethicml.vision import TorchImageDataset

from ..dataset import Dataset

__all__ = ["CelebAttrs", "celeba", "create_celeba_dataset"]


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
        "0B7EVK8r0v71pZjFTYXZWM3FlRnM",  # File ID
        "00d2c5bc6d35e252742224ab0c1e8fcb",  # MD5 Hash
        "img_align_celeba.zip",  # Filename
    ),
    ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt",),
    (
        "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
        "d32c9cbf5e040fd4025c592c306e6668",
        "list_eval_partition.txt",
    ),
]


def celeba(
    download_dir: str,
    label: CelebAttrs = "Smiling",
    sens_attr: CelebAttrs = "Male",
    download: bool = False,
    check_integrity: bool = True,
) -> Tuple[Optional[Dataset], Path]:
    """Get CelebA dataset."""
    root = Path(download_dir)
    if True:  # pylint: disable=using-constant-test
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
        discrete_features: List[str] = []
        for group in disc_feature_groups.values():
            discrete_features += group
        assert sens_attr in discrete_features
        assert label in discrete_features
        continuous_features = ["filename"]

    img_dir = root / _BASE_FOLDER / "img_align_celeba"
    if download:
        _download(root)
    elif check_integrity:
        if not _check_integrity(root):
            return None, img_dir

    dataset_obj = Dataset(
        name=f"CelebA, s={sens_attr}, y={label}",
        sens_attrs=[sens_attr],
        s_prefix=[sens_attr],
        class_labels=[label],
        class_label_prefix=[label],
        disc_feature_groups=disc_feature_groups,
        features=discrete_features + continuous_features,
        cont_features=continuous_features,
        num_samples=202599,
        filename_or_path="celeba.csv.zip",
        discrete_only=False,
    )
    return dataset_obj, img_dir


def _check_integrity(base: Path) -> bool:
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


def create_celeba_dataset(
    root: str,
    biased: bool,
    mixing_factor: float,
    unbiased_pcnt: float,
    sens_attr_name: CelebAttrs,
    target_attr_name: CelebAttrs,
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
    sens_attr_name = cast(CelebAttrs, sens_attr_name.capitalize())
    target_attr_name = cast(CelebAttrs, target_attr_name.capitalize())

    dataset, base_dir = celeba(
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
            data=biased_dt,
            root=base_dir,
            map_to_binary=True,
            transform=transform,
            target_transform=target_transform,
        )
    else:
        return TorchImageDataset(
            data=unbiased_dt,
            root=base_dir,
            map_to_binary=True,
            transform=transform,
            target_transform=target_transform,
        )
