"""Class to describe features of the Test dataset."""
from typing import List, Dict
from .dataset import Dataset


class Toy(Dataset):
    """Dataset with toy data for testing."""

    cont_features: List[str]
    disc_features: List[str]

    def __init__(self) -> None:
        """Init Toy dataset."""
        super().__init__()
        self.cont_features = ["a1", "a2"]
        self.disc_features = []

    @property
    def name(self) -> str:
        """Get dataset name."""
        return "Toy"

    @property
    def filename(self) -> str:
        """Get filename."""
        return "toy.csv"

    @property
    def feature_split(self) -> Dict[str, List[str]]:
        """Get split of the features."""
        return {"x": ["a1", "a2"], "s": ["s"], "y": ["y"]}

    def __len__(self) -> int:
        """Length of the dataset."""
        return 2000
