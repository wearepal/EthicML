"""Test Hydra compatability."""
from typing import Type

from omegaconf import OmegaConf
import pytest

from ethicml import models


@pytest.mark.parametrize(
    "algo_class",
    [
        models.Agarwal,
        models.Blind,
        models.Corels,
        models.DPOracle,
        models.LR,
        models.LRCV,
        models.Majority,
        models.Oracle,
        models.SVM,
        models.DRO,
        models.Reweighting,
        models.MLP,
    ],
)
def test_hydra_compatibility(algo_class: Type[models.InAlgorithm]) -> None:
    """Test hydra compatibility."""
    # create config object from dataclass (usually taken care of by hydra)
    conf = OmegaConf.structured(algo_class)
    assert not hasattr(conf, "is_fairness_algo")  # this attribute should not be configurable

    # instantiate object from the config
    model = OmegaConf.to_object(conf)
    assert isinstance(model, algo_class)
    assert isinstance(model.is_fairness_algo, bool)


@pytest.mark.parametrize("algo_class,hidden_attr", [(models.Reweighting, "group_weights")])
def test_dont_leak_impl_detail(algo_class: Type[models.InAlgorithm], hidden_attr: str) -> None:
    conf = OmegaConf.structured(algo_class)
    assert not hasattr(conf, hidden_attr)
