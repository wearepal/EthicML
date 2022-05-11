"""Class to describe features of the Synthetic dataset."""
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Type

import teext as tx

from ..dataset import LoadableDataset

__all__ = ["SyntheticScenarios", "SyntheticTargets", "Synthetic"]


class SyntheticScenarios(Enum):
    """Scenarios for the synthetic dataset."""

    S1 = 1
    S2 = 2
    S3 = 3
    S4 = 4


class SyntheticTargets(Enum):
    """Targets for the synthetic dataset."""

    Y1 = 1
    Y2 = 2
    Y3 = 3


@dataclass
class Synthetic(LoadableDataset):
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

    Scenarios: ClassVar[Type[SyntheticScenarios]] = SyntheticScenarios
    Targets: ClassVar[Type[SyntheticTargets]] = SyntheticTargets

    scenario: SyntheticScenarios = SyntheticScenarios.S1
    target: SyntheticTargets = SyntheticTargets.Y3
    fair: bool = False
    num_samples: int = 1_000

    def __post_init__(self) -> None:
        scenario: int = self.scenario.value
        target: int = self.target.value
        assert self.num_samples <= 100_000
        num_samples = tx.assert_positive_int(self.num_samples)

        super().__init__(
            name=f"Synthetic - Scenario {scenario}, target {target}"
            + (" fair" if self.fair else ""),
            num_samples=num_samples,
            filename_or_path=f"synthetic_scenario_{scenario}.csv",
            features=["x1f", "x2f", "n1", "n2"] if self.fair else ["x1", "x2", "n1", "n2"],
            cont_features=["x1f", "x2f", "n1", "n2"] if self.fair else ["x1", "x2", "n1", "n2"],
            sens_attr_spec="s",
            class_label_spec=f"y{target}" + ("f" if self.fair else ""),
            discrete_only=self.discrete_only,
        )
