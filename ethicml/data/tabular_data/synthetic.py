"""Class to describe features of the Synthetic dataset."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, List, Tuple, Type, Union

from ranzen import implements

from ethicml.data.dataset import CSVDatasetDC, LabelSpecsPair
from ethicml.data.util import DiscFeatureGroup, single_col_spec

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
class Synthetic(CSVDatasetDC):
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
        assert 0 < self.num_samples <= 100_000

    @implements(CSVDatasetDC)
    def get_cont_features(self) -> List[str]:
        return ["x1f", "x2f", "n1", "n2"] if self.fair else ["x1", "x2", "n1", "n2"]

    @implements(CSVDatasetDC)
    def get_name(self) -> str:
        return (
            f"Synthetic - Scenario {self.scenario.value}, "
            f"target {self.target.value}{' fair' if self.fair else ''}"
        )

    @implements(CSVDatasetDC)
    def get_label_specs(self) -> Tuple[LabelSpecsPair, List[str]]:
        y = single_col_spec(f"y{self.target.value}{'f' if self.fair else ''}")
        return LabelSpecsPair(s=single_col_spec("s"), y=y), []

    @implements(CSVDatasetDC)
    def get_num_samples(self) -> int:
        return self.num_samples

    @implements(CSVDatasetDC)
    def get_filename_or_path(self) -> Union[str, Path]:
        return f"synthetic_scenario_{self.scenario.value}.csv"

    @implements(CSVDatasetDC)
    def get_unfiltered_disc_feat_groups(self) -> DiscFeatureGroup:
        return {}
