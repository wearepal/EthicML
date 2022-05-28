from enum import Enum
from typing import Type, Union

import pytest
from omegaconf import OmegaConf, ValidationError

import ethicml as em


@pytest.mark.parametrize(
    "data_class,value,should_pass",
    [
        (em.Admissions, em.Admissions.Splits.GENDER, True),
        (em.Adult, "Race", False),
        (em.Adult, em.Adult.Splits.RACE, True),
        (em.Compas, em.Compas.Splits.RACE_SEX, True),
        (em.Credit, em.Credit.Splits.SEX, True),
        (em.Crime, em.Crime.Splits.RACE_BINARY, True),
        (em.German, em.German.Splits.SEX, True),
        (em.Health, em.Health.Splits.SEX, True),
        (em.Law, em.Law.Splits.SEX, True),
        (em.Sqf, em.Sqf.Splits.RACE_SEX, True),
    ],
)
def test_datasets_with_split(
    data_class: Type[em.Dataset], value: Union[str, Enum], should_pass: bool
) -> None:

    # This will fail if the supplied `data_class` has types other than bool, int, float, enum, str.
    # OmegaConf is what hydra uses internally.
    conf = OmegaConf.structured(data_class)

    if should_pass:
        conf.split = value
    else:
        # OmegaConf raises a ValidationError if the assigned value does not match the delcared type.
        with pytest.raises(ValidationError):
            conf.split = value


@pytest.mark.parametrize("data_class", [em.Synthetic, em.Toy, em.NonBinaryToy, em.Lipton])
def test_datasets_without_split(data_class: Type[em.Dataset]) -> None:
    # This will fail if the supplied `data_class` has types other than bool, int, float, enum, str.
    # OmegaConf is what hydra uses internally.
    OmegaConf.structured(data_class)
