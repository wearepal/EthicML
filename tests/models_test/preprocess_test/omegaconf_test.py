from typing import Type

import pytest
from omegaconf import OmegaConf

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
