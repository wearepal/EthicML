from typing import Type

from omegaconf import OmegaConf
import pytest

from ethicml import models


@pytest.mark.parametrize(
    "algo_class", [models.Upsampler, models.Beutel, models.Calders, models.VFAE, models.Zemel]
)
def test_hydra_compatibility(algo_class: Type[models.PreAlgorithm]) -> None:
    # create config object from dataclass (usually taken care of by hydra)
    conf = OmegaConf.structured(algo_class)

    # instantiate object from the config
    model = OmegaConf.to_object(conf)
    assert isinstance(model, algo_class)


@pytest.mark.parametrize(
    ("algo_class", "hidden_attr"),
    [(models.Upsampler, "_out_size"), (models.Calders, "_out_size"), (models.Zemel, "_in_size")],
)
def test_dont_leak_impl_detail(algo_class: Type[models.PreAlgorithm], hidden_attr: str) -> None:
    conf = OmegaConf.structured(algo_class)
    assert not hasattr(conf, hidden_attr)
