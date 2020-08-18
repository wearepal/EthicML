"""Class to describe features of the toy dataset with more than 2 classes."""
from ..dataset import Dataset

__all__ = ["nonbinary_toy"]


def nonbinary_toy() -> Dataset:
    """Dataset with non-binary toy data for testing."""
    return Dataset(
        name="NonBinaryToy",
        num_samples=100,
        filename_or_path="non-binary-toy.csv",
        features=["x1", "x2"],
        cont_features=["x1", "x2"],
        sens_attr_spec="sens",
        class_label_spec="label",
        discrete_only=False,
    )
