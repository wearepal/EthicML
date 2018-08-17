"""
Class to that can be used to create
"""
from typing import Dict, List

from ethicml.data.dataset import Dataset


class ConfigurableDataset(Dataset):
    def __init__(self):
        self.filename: str = "No filename assigned. " \
                             "Use set_filename(<filename>)"
        self.feature_split: Dict[str, List[str]] = {}

    def get_filename(self) -> str:
        return self.filename

    def set_filename(self, filename: str) -> None:
        self.filename = filename
        return

    def get_feature_split(self) -> Dict[str, List[str]]:
        return self.feature_split

    def set_feature_split(self, feature_split: Dict[str, List[str]]) -> None:
        self.feature_split = feature_split
        return
