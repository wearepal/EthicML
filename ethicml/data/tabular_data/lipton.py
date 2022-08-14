"""Class to describe features of the Lipton dataset."""
from __future__ import annotations
from dataclasses import dataclass

from ..dataset import LegacyDataset

__all__ = ["Lipton"]


@dataclass
class Lipton(LegacyDataset):
    """Synthetic dataset from the Lipton et al. 2018.

    Described in section 4.1 of `Does mitigating ML's impact disparity require treatment disparity?`

    .. code-block:: bibtex

        @article{lipton2018does,
            title={Does mitigating ML's impact disparity require treatment disparity?},
            author={Lipton, Zachary and McAuley, Julian and Chouldechova, Alexandra},
            journal={Advances in neural information processing systems},
            volume={31},
            pages={8125--8135},
            year={2018}
        }
    """

    def __post_init__(self) -> None:
        continuous_features = ["hair_length", "work_experience"]
        super().__init__(
            name="Lipton",
            num_samples=2_000,
            filename_or_path="lipton.csv",
            features=continuous_features,
            cont_features=continuous_features,
            sens_attr_spec="sens",
            class_label_spec="hired",
        )
