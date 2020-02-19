"""Class to describe features of the toy dataset with more than 2 classes."""
from typing import Dict, List

from ethicml.common import implements

from .dataset import Dataset


class NonBinaryToy(Dataset):
    """Dataset with non-binary toy data for testing."""

    cont_features: List[str]
    disc_features: List[str]

    def __init__(self) -> None:
        """Init Toy dataset with more than 2 classes."""
        super().__init__()
        self.cont_features = ["x1", "x2"]
        self.disc_features = []

    @property
    def name(self) -> str:
        """Getter for dataset name."""
        return "NonBinaryToy"

    @property
    def filename(self) -> str:
        """Getter for filename."""
        return "non-binary-toy.csv"

    @property
    def feature_split(self) -> Dict[str, List[str]]:
        """Get the split of the dataset features."""
        return {"x": ["x1", "x2"], "s": ["sens"], "y": ["label"]}

    @implements(Dataset)
    def __len__(self) -> int:
        return 100
