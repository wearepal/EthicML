"""Class to describe features of the Test dataset."""
from warnings import warn

from ..dataset import Dataset

__all__ = ["Toy", "toy"]


def Toy() -> Dataset:  # pylint: disable=invalid-name
    """Dataset with toy data for testing."""
    warn("The Toy class is deprecated. Use the function instead.", DeprecationWarning)
    return toy()


def toy() -> Dataset:
    """Dataset with toy data for testing."""
    return Dataset(
        name="Toy",
        num_samples=2000,
        filename_or_path="toy.csv",
        features=["a1", "a2"],
        cont_features=["a1", "a2"],
        sens_attrs=["s"],
        class_labels=["y"],
        discrete_only=False,
    )
