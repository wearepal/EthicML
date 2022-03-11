"""Class to describe features of the Toy dataset."""

from dataclasses import dataclass

from ..dataset import LoadableDataset
from ..util import flatten_dict

__all__ = ["Toy", "toy"]


def toy() -> "Toy":
    """Dataset with toy data for testing."""
    return Toy()


@dataclass
class Toy(LoadableDataset):
    """Dataset with toy data for testing."""

    def __post_init__(self) -> None:
        disc_feature_groups = {
            "disc_1": ["disc_1_a", "disc_1_b", "disc_1_c", "disc_1_d", "disc_1_e"],
            "disc_2": ["disc_2_x", "disc_2_y", "disc_2_z"],
        }
        discrete_features = flatten_dict(disc_feature_groups)
        continuous_features = ["a1", "a2"]
        super().__init__(
            name="Toy",
            num_samples=400,
            filename_or_path="toy.csv",
            features=continuous_features + discrete_features,
            cont_features=continuous_features,
            sens_attr_spec="sensitive-attr",
            class_label_spec="decision",
            discrete_only=False,
            discrete_feature_groups=disc_feature_groups,
        )
