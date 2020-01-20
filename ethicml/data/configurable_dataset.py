"""Class to that can be used to wrap your own dataset in a way EthicML can understand."""
from typing import Dict, List

from ethicml.common import implements
from .dataset import Dataset


class ConfigurableDataset(Dataset):
    """Dataset that is configurable."""

    def __init__(self) -> None:
        """Init configurable datset object."""
        self.filename = ""
        super().__init__()
        self._dataname: str = "No name assigned."
        self._filename: str = ("No filename assigned. " "Use set_filename(<filename>)")
        self._feature_split: Dict[str, List[str]] = {}
        self.cont_features: List[str] = []
        self.disc_features: List[str] = []

    @property
    def name(self) -> str:
        """Getter for the dataset name."""
        return self._dataname

    @name.setter
    def name(self, value: str) -> None:
        """Setter for the dataset name."""
        self._dataname = value

    @property
    def filename(self) -> str:
        """Getter for the filename."""
        return self._filename

    @filename.setter
    def filename(self, value: str) -> None:
        """Setter for the filename."""
        self._filename = value

    @property
    def feature_split(self) -> Dict[str, List[str]]:
        """Getter for split of features."""
        return self._feature_split

    @feature_split.setter
    def feature_split(self, value: Dict[str, List[str]]) -> None:
        """Setter for split of features."""
        self._feature_split = value

    @implements(Dataset)
    def __len__(self) -> int:
        return 0
