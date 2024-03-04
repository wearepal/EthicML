"""Test compatibility of the dataset objects with OmegaConf."""

from enum import Enum
from typing import Type, Union

from omegaconf import OmegaConf, ValidationError
import pytest

import ethicml.data as emda


@pytest.mark.parametrize(
    ("data_class", "value", "should_pass"),
    [
        (emda.Admissions, emda.Admissions.Splits.GENDER, True),
        (emda.Adult, "Race", False),
        (emda.Adult, emda.Adult.Splits.RACE, True),
        (emda.Compas, emda.Compas.Splits.RACE_SEX, True),
        (emda.Credit, emda.Credit.Splits.SEX, True),
        (emda.Crime, emda.Crime.Splits.RACE_BINARY, True),
        (emda.German, emda.German.Splits.SEX, True),
        (emda.Health, emda.Health.Splits.SEX, True),
        (emda.Law, emda.Law.Splits.SEX, True),
        (emda.Sqf, emda.Sqf.Splits.RACE_SEX, True),
    ],
)
def test_datasets_with_split(
    data_class: Type[emda.Dataset],
    value: Union[str, Enum],
    should_pass: bool,  # noqa: FBT001
) -> None:
    """Test datasets with split."""
    # This will fail if the supplied `data_class` has types other than bool, int, float, enum, str.
    # OmegaConf is what hydra uses internally.
    conf = OmegaConf.structured(data_class)

    if should_pass:
        conf.split = value
    else:
        # OmegaConf raises a ValidationError if the assigned value does not match the delcared type.
        with pytest.raises(ValidationError):
            conf.split = value


@pytest.mark.parametrize("data_class", [emda.Synthetic, emda.Toy, emda.NonBinaryToy, emda.Lipton])
def test_datasets_without_split(data_class: Type[emda.Dataset]) -> None:
    """Test datasets without split."""
    # This will fail if the supplied `data_class` has types other than bool, int, float, enum, str.
    # OmegaConf is what hydra uses internally.
    OmegaConf.structured(data_class)
