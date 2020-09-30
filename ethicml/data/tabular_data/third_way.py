"""Class to describe features of the Synthetic dataset."""
from warnings import warn

from ..dataset import Dataset

__all__ = ["ThirdWay", "third_way"]


def ThirdWay() -> Dataset:  # pylint: disable=invalid-name
    """Dataset with synthetic scenario 1 data."""
    warn("The Synthetic class is deprecated. Use the function instead.", DeprecationWarning)
    return third_way()


def third_way() -> Dataset:
    r"""Dataset with synthetic data."""
    return Dataset(
        name=f"ThirdWay",
        num_samples=5_000,
        filename_or_path=f"3rd.csv",
        features=["x_0", "x_1", "x_2", "x_3", "x_4", "x_5", "sens", "y", "y_bar"],
        cont_features=[],
        sens_attr_spec="sens",
        s_prefix="sens",
        class_label_spec=f"y",
        class_label_prefix="y",
        discrete_only=False,
    )
