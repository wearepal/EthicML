"""Test whether metrics classes are compatible with OmegaConf and thus hydra."""
from typing import Type

from omegaconf import OmegaConf
import pytest

from ethicml import metrics


@pytest.mark.parametrize(
    "metric_class",
    [
        metrics.AS,
        metrics.AbsCV,
        metrics.Accuracy,
        metrics.AverageOddsDiff,
        metrics.BCR,
        metrics.BalancedAccuracy,
        metrics.CV,
        metrics.F1,
        metrics.FNR,
        metrics.FPR,
        metrics.Hsic,
        metrics.NMI,
        metrics.NPV,
        metrics.PPV,
        metrics.ProbNeg,
        metrics.ProbOutcome,
        metrics.ProbPos,
        metrics.RenyiCorrelation,
        metrics.TNR,
        metrics.TPR,
        metrics.Theil,
    ],
)
def test_omegaconf(metric_class: Type[metrics.Metric]) -> None:
    """Test metric classes with OmegaConf."""
    # This will fail if the supplied `data_class` has types other than bool, int, float, enum, str.
    # OmegaConf is what hydra uses internally.
    OmegaConf.structured(metric_class)
