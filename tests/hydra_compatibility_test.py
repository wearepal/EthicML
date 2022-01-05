from enum import Enum
from typing import Type, Union

import pytest
from omegaconf import OmegaConf, ValidationError

from ethicml import Adult, AdultSplits, Compas, CompasSplits, Dataset


@pytest.mark.parametrize(
    "data_class,value,should_pass",
    [
        (Adult, AdultSplits.RACE, True),
        (Adult, "Race", False),
        (Compas, CompasSplits.RACE_SEX, True),
    ],
)
def test_datasets_with_split(
    data_class: Type[Dataset], value: Union[str, Enum], should_pass: bool
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
