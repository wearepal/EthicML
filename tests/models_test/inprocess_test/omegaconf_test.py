from typing import Type

import pytest
from omegaconf import OmegaConf

import ethicml as em


@pytest.mark.parametrize(
    "algo_class",
    [
        em.Agarwal,
        em.Blind,
        em.Corels,
        em.DPOracle,
        em.LR,
        em.LRCV,
        em.LRProb,
        em.Majority,
        em.Oracle,
        em.SVM,
    ],
)
def test_logistic_regression(algo_class: Type[em.InAlgorithm]) -> None:
    # create config object from dataclass (usually taken care of by hydra)
    conf = OmegaConf.structured(algo_class)
    # set values
    conf.seed = 12
    assert not hasattr(conf, "is_fairness_algo")  # this attribute should not be configurable

    # instantiate object from the config
    model = OmegaConf.to_object(conf)
    assert isinstance(model, algo_class)
    assert model.seed == conf.seed
    assert isinstance(model.is_fairness_algo, bool)
