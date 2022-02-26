from typing import Type
import pytest
from omegaconf import OmegaConf

import ethicml as em


@pytest.mark.parametrize("algo_class", [em.LR, em.Majority, em.Corels, em.Oracle, em.DPOracle])
def test_logistic_regression(algo_class: Type[em.InAlgorithm]) -> None:
    # create config object from dataclass (usually taken care of by hydra)
    conf = OmegaConf.structured(algo_class)
    # set values
    conf.seed = 12
    assert not hasattr(conf, "is_fairness_algo")  # this attribute should not be configurable

    # instantiate object from the config
    lr = OmegaConf.to_object(conf)
    assert isinstance(lr, algo_class)
    assert lr.seed == conf.seed
    assert isinstance(lr.is_fairness_algo, bool)
