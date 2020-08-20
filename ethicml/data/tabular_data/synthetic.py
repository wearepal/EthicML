"""Class to describe features of the Synthetic dataset."""
from warnings import warn

from typing_extensions import Literal

from ..dataset import Dataset

__all__ = ["Synthetic", "synthetic"]


def Synthetic() -> Dataset:  # pylint: disable=invalid-name
    """Dataset with synthetic scenario 1 data."""
    warn("The Synthetic class is deprecated. Use the function instead.", DeprecationWarning)
    return synthetic()


def synthetic(scenario: Literal[1, 2, 3] = 1, target: Literal[1, 2, 3] = 1) -> Dataset:
    r"""Dataset with synthetic data.

    ⊥ = is independent of
    ~ = is an ancestor of in the causal model used to generate the data

    Scenario 1 = X⊥S & Y⊥S.
    Scenario 2 = X_2⊥S & Y_2⊥S; X_1~S, Y_1~S & Y_3~S
    Scenario 3 = X⊥S, Y_1⊥S, Y_2⊥S; Y_3~S
    """
    assert scenario in [1, 2, 3]
    assert target in [1, 2, 3]

    return Dataset(
        name=f"Synthetic - Scenario {scenario}, target {target}",
        num_samples=1000,
        filename_or_path=f"synthetic_scenario_{scenario}.csv",
        features=["x1", "x2"],
        cont_features=["x1", "x2"],
        sens_attr_spec="s",
        class_label_spec=f"y{target}",
        discrete_only=False,
    )
