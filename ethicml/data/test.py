"""
Class to describe features of the Test dataset
"""
from typing import List, Dict
from .dataset import Dataset


class Test(Dataset):
    """Class to describe features of the Test dataset"""
    cont_features: List[str]
    disc_features: List[str]

    def __init__(self):
        super().__init__()
        self.cont_features = ["a1", "a2"]
        self.disc_features = []

    @property
    def name(self) -> str:
        return "Test"

    @property
    def filename(self) -> str:
        return "test.csv"

    @property
    def feature_split(self) -> Dict[str, List[str]]:
        return {
            "x": ["a1", "a2"],
            "s": ["s"],
            "y": ["y"]
        }
