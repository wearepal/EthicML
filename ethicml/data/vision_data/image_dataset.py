"""Base class for image datasets."""
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List

from ethicml.common import implements
from ethicml.data.tabular_data.dataset import Dataset

__all__ = ["ImageDataset"]


class ImageDataset(Dataset):
    """Base class for image datasets."""

    _disc_feature_groups: Dict[str, List[str]]

    def __init__(
        self,
        name: str,
        disc_feature_groups: Dict[str, List[str]],
        continuous_features: List[str],
        length: int,
        csv_file: str,
        img_dir: Path,
    ):
        """Init image dataset."""
        # these have to be set before the super call
        self.__name = name
        self.__len = length
        self.__filename = csv_file
        self.__img_dir = img_dir
        super().__init__(disc_feature_groups=disc_feature_groups)

        discrete_features: List[str] = []
        for group in disc_feature_groups.values():
            discrete_features += group

        self.continuous_features = continuous_features
        self.features = self.continuous_features + discrete_features

    @property
    def name(self) -> str:
        """Getter for dataset name."""
        return self.__name

    @property
    def filename(self) -> str:
        """Getter for filename."""
        return self.__filename

    @implements(Dataset)
    def __len__(self) -> int:
        return self.__len

    @abstractmethod
    def check_integrity(self) -> bool:
        """Check integrity of the data folder.

        Returns:
            bool: Boolean indicating whether the file containing the celeba data
                  is readable.
        """

    @abstractmethod
    def download(self) -> None:
        """Attempt to download data if files cannot be found in the base folder."""

    @property
    def img_dir(self) -> Path:
        """Directory where the images are stored."""
        return self.__img_dir
