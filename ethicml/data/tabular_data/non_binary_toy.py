"""Class to describe features of the toy dataset with more than 2 classes."""
from dataclasses import dataclass

from ..dataset import LoadableDataset

__all__ = ["nonbinary_toy", "NonBinaryToy"]


def nonbinary_toy() -> "NonBinaryToy":
    """Dataset with non-binary toy data for testing."""
    return NonBinaryToy()


@dataclass
class NonBinaryToy(LoadableDataset):
    """Dataset with non-binary toy data for testing."""

    def __post_init__(self) -> None:
        super().__init__(
            name="NonBinaryToy",
            num_samples=100,
            filename_or_path="non-binary-toy.csv",
            features=["x1", "x2"],
            cont_features=["x1", "x2"],
            sens_attr_spec="sens",
            class_label_spec="label",
            discrete_only=self.discrete_only,
        )
