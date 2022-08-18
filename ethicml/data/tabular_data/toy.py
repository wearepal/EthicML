"""Class to describe features of the Toy dataset."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ..dataset import LabelSpecsPair, StaticCSVDataset
from ..util import DiscFeatureGroup, single_col_spec

__all__ = ["Toy"]


@dataclass
class Toy(StaticCSVDataset):
    """Dataset with toy data for testing."""

    num_samples: ClassVar[int] = 400
    csv_file: ClassVar[str] = "toy.csv"

    @property  # type: ignore[misc]
    @implements(StaticCSVDataset)
    def name(self) -> str:
        return "Toy"

    @implements(StaticCSVDataset)
    def get_label_specs(self) -> LabelSpecsPair:
        return LabelSpecsPair(
            s=single_col_spec("sensitive-attr"),
            y=single_col_spec("decision"),
            to_remove=[],
        )

    @property  # type: ignore[misc]
    @implements(StaticCSVDataset)
    def unfiltered_disc_feat_groups(self) -> DiscFeatureGroup:
        return {
            "disc_1": ["disc_1_a", "disc_1_b", "disc_1_c", "disc_1_d", "disc_1_e"],
            "disc_2": ["disc_2_x", "disc_2_y", "disc_2_z"],
        }

    @property  # type: ignore[misc]
    @implements(StaticCSVDataset)
    def continuous_features(self) -> list[str]:
        return ["a1", "a2"]
