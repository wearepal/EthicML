"""Class to describe features of the Synthetic dataset."""

from typing_extensions import Literal

import teext as tx

from ..dataset import Dataset

__all__ = ["synthetic", "Synthetic"]


def synthetic(
    scenario: Literal[1, 2, 3, 4] = 1,
    target: Literal[1, 2, 3] = 3,
    fair: bool = False,
    num_samples: int = 1_000,
) -> Dataset:
    r"""Dataset with synthetic data.

    ⊥ = is independent of
    ~ = is an ancestor of in the causal model used to generate the data

    Scenario 1 = X⊥S & Y⊥S.
        - This models completely fair data.
    Scenario 2 = X_2⊥S & Y_2⊥S; X_1~S, Y_1~S & Y_3~S
        - This models data where the inputs are biased. This is propogated through to the target.
    Scenario 3 = X⊥S, Y_1⊥S, Y_2⊥S; Y_3~S
        - This models data where the target is biased.
    Scenario 4 = X_2⊥S, Y_2⊥S; X_1~S, Y_1~S, Y_3~S
        - This models data where both the input and target are directly biased.
    """
    return Synthetic(scenario=scenario, target=target, fair=fair, num_samples=num_samples)


class Synthetic(Dataset):
    r"""Dataset with synthetic data.

    ⊥ = is independent of
    ~ = is an ancestor of in the causal model used to generate the data

    Scenario 1 = X⊥S & Y⊥S.
        - This models completely fair data.
    Scenario 2 = X_2⊥S & Y_2⊥S; X_1~S, Y_1~S & Y_3~S
        - This models data where the inputs are biased. This is propogated through to the target.
    Scenario 3 = X⊥S, Y_1⊥S, Y_2⊥S; Y_3~S
        - This models data where the target is biased.
    Scenario 4 = X_2⊥S, Y_2⊥S; X_1~S, Y_1~S, Y_3~S
        - This models data where both the input and target are directly biased.
    """

    def __init__(
        self,
        scenario: Literal[1, 2, 3, 4] = 1,
        target: Literal[1, 2, 3] = 3,
        fair: bool = False,
        num_samples: int = 1_000,
    ):
        assert scenario in [1, 2, 3, 4]
        assert target in [1, 2, 3]
        assert num_samples <= 100_000
        num_samples = tx.assert_positive_int(num_samples)

        super().__init__(
            name=f"Synthetic - Scenario {scenario}, target {target}" + (" fair" if fair else ""),
            num_samples=num_samples,
            filename_or_path=f"synthetic_scenario_{scenario}.csv",
            features=["x1f", "x2f", "n1", "n2"] if fair else ["x1", "x2", "n1", "n2"],
            cont_features=["x1f", "x2f", "n1", "n2"] if fair else ["x1", "x2", "n1", "n2"],
            sens_attr_spec="s",
            class_label_spec=f"y{target}" + ("f" if fair else ""),
            discrete_only=False,
        )
