"""Class to describe features of the toy dataset with more than 2 classes."""
from __future__ import annotations
from dataclasses import dataclass

from ..dataset import LegacyDataset
from ..util import flatten_dict

__all__ = ["NonBinaryToy"]


@dataclass
class NonBinaryToy(LegacyDataset):
    """Dataset with toy data for testing."""

    def __post_init__(self) -> None:
        disc_feature_groups = {
            # "decision": [f"decision_{i}" for i in range(5)],
            "sensitive-attr": [f"sensitive-attr_{i}" for i in range(2)],
            "disc_1": ["disc_1_a", "disc_1_b", "disc_1_c", "disc_1_d", "disc_1_e"],
            "disc_2": ["disc_2_x", "disc_2_y", "disc_2_z"],
        }
        discrete_features = flatten_dict(disc_feature_groups)
        continuous_features = ["a1", "a2"]
        super().__init__(
            name="NonBinaryToy",
            num_samples=400,
            filename_or_path="nbt.csv",
            features=continuous_features + discrete_features,
            cont_features=continuous_features,
            sens_attr_spec="sensitive-attr_1",
            s_feature_groups=["sensitive-attr"],
            class_label_spec="decision",
            discrete_feature_groups=disc_feature_groups,
        )
