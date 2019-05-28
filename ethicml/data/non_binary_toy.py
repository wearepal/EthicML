"""
Class to describe features of the toy dataset with more than 2 classes
"""
from typing import List, Dict
from .dataset import Dataset


class NonBinaryToy(Dataset):
    """Class to describe features of the Non-Binary toy dataset"""

    cont_features: List[str]
    disc_features: List[str]

    def __init__(self):
        super().__init__()
        self.cont_features = ["x1", "x2"]
        self.disc_features = []

    @property
    def name(self) -> str:
        return "NonBinaryToy"

    @property
    def filename(self) -> str:
        return "non-binary-toy.csv"

    @property
    def feature_split(self) -> Dict[str, List[str]]:
        return {"x": ["x1", "x2"], "s": ["sens"], "y": ["label"]}
