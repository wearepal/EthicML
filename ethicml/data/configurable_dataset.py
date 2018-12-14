"""
Class to that can be used to create
"""
from typing import Dict, List
from .dataset import Dataset


class ConfigurableDataset(Dataset):
    def __init__(self):
        self.name: str = "No name assigned."
        self.filename: str = "No filename assigned. " \
                             "Use set_filename(<filename>)"
        self.feature_split: Dict[str, List[str]] = {}

    def get_dataset_name(self) -> str:
        return self.name

    def set_dataset_name(self, name: str) -> None:
        self.name = name

    def get_filename(self) -> str:
        return self.filename

    def set_filename(self, filename: str) -> None:
        self.filename = filename

    def get_feature_split(self) -> Dict[str, List[str]]:
        return self.feature_split

    def set_feature_split(self, feature_split: Dict[str, List[str]]) -> None:
        self.feature_split = feature_split
