"""Test whether metrics classes are compatible with OmegaConf and thus hydra."""
from typing import Type
from omegaconf import OmegaConf
import pytest

import ethicml as em


@pytest.mark.parametrize(
    "metric_class",
    [
        em.AS,
        em.AbsCV,
        em.Accuracy,
        em.AverageOddsDiff,
        em.BCR,
        em.BalancedAccuracy,
        em.CV,
        em.F1,
        em.FNR,
        em.FPR,
        em.Hsic,
        em.NMI,
        em.NPV,
        em.PPV,
        em.ProbNeg,
        em.ProbOutcome,
        em.ProbPos,
        em.RenyiCorrelation,
        em.TNR,
        em.TPR,
        em.Theil
    ],
)
def test_omegaconf(metric_class: Type[em.Metric]):
    # This will fail if the supplied `data_class` has types other than bool, int, float, enum, str.
    # OmegaConf is what hydra uses internally.
    OmegaConf.structured(metric_class)
