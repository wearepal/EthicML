from typing import Type

import pytest
from omegaconf import OmegaConf

import ethicml as em


@pytest.mark.parametrize("algo_class", [em.Upsampler, em.Beutel, em.Calders, em.VFAE, em.Zemel])
def test_hydra_compatibility(algo_class: Type[em.PreAlgorithm]) -> None:
    # create config object from dataclass (usually taken care of by hydra)
    conf = OmegaConf.structured(algo_class)

    # instantiate object from the config
    model = OmegaConf.to_object(conf)
    assert isinstance(model, algo_class)
