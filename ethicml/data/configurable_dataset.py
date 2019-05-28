"""
Class to that can be used to create
"""
from typing import Dict, List
from .dataset import Dataset


class ConfigurableDataset(Dataset):
    """Dataset that is configurable"""

    def __init__(self):
        self.filename = ""
        super().__init__()
        self._dataname: str = "No name assigned."
        self._filename: str = ("No filename assigned. " "Use set_filename(<filename>)")
        self._feature_split: Dict[str, List[str]] = {}
        self.cont_features: List[str] = []
        self.disc_features: List[str] = []

    @property
    def name(self) -> str:
        return self._dataname

    @name.setter
    def name(self, value: str) -> None:
        self._dataname = value

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, value: str) -> None:
        self._filename = value

    @property
    def feature_split(self) -> Dict[str, List[str]]:
        return self._feature_split

    @feature_split.setter
    def feature_split(self, value: Dict[str, List[str]]) -> None:
        self._feature_split = value
