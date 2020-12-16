"""Class to describe features of the Toy dataset."""

from ..dataset import Dataset
from ..util import deprecated, flatten_dict

__all__ = ["Toy", "toy"]


@deprecated
def Toy() -> Dataset:  # pylint: disable=invalid-name
    """Dataset with toy data for testing."""
    return toy()


def toy() -> Dataset:
    """Dataset with toy data for testing."""
    disc_feature_groups = {
        "disc_1": ["disc_1_a", "disc_1_b", "disc_1_c", "disc_1_d", "disc_1_e"],
        "disc_2": ["disc_2_x", "disc_2_y", "disc_2_z"],
    }
    discrete_features = flatten_dict(disc_feature_groups)
    continuous_features = ["a1", "a2"]
    return Dataset(
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
