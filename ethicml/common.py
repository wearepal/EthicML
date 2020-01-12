"""Common variables / consts that make things run smoother."""

import os
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

ROOT_DIR: str = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
ROOT_PATH: Path = Path(__file__).parent.resolve()

_F = TypeVar('_F', bound=Callable[..., Any])


class implements:  # pylint: disable=invalid-name
    """Mark a function as implementing an interface."""

    def __init__(self, interface: Type):
        """Instantiate the decorator.

        Args:
            interface: the interface that is implemented
        """
        self.interface = interface

    def __call__(self, func: _F) -> _F:
        """Take a function and return it unchanged."""
        return func
