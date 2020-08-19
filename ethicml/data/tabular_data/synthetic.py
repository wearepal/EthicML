"""Class to describe features of the Synthetic dataset."""
from warnings import warn

from typing_extensions import Literal

from ..dataset import Dataset

__all__ = ["Synthetic", "synthetic"]


def Synthetic() -> Dataset:  # pylint: disable=invalid-name
    """Dataset with synthetic scenario 1 data."""
    warn("The Synthetic class is deprecated. Use the function instead.", DeprecationWarning)
    return synthetic()


def synthetic(scenario: Literal[1] = 1, target: Literal[1, 2, 3] = 1) -> Dataset:
    r"""Dataset with synthetic data.

    Scenario 1 = X⊥S & Y⊥S.
    """
    if scenario == 1:
        return Dataset(
            name=f"Synthetic - Scenario {scenario}, target {target}",
            num_samples=1000,
            filename_or_path="synthetic_scenario_1.csv",
            features=["x1", "x2"],
            cont_features=["x1", "x2"],
            sens_attr_spec="s",
            class_label_spec=f"y{target}",
            discrete_only=False,
        )
    else:
        raise NotImplementedError("We'll have more scenarios available soon!")
