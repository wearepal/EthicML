"""Common variables / consts that make things run smoother."""

import os
from pathlib import Path
from typing import Callable, Any, TypeVar, Type

ROOT_DIR: str = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
ROOT_PATH: Path = Path(__file__).parent.resolve()

FuncType = Callable[..., Any]
_F = TypeVar('_F', bound=FuncType)


class implements:  # pylint: disable=invalid-name
    """Mark a function as implementing an interface."""

    def __init__(self, class_: Type):
        """Instantiate this decorator.

        Args:
            class_: the interface which we're implementing
        """
        self.class_ = class_

    def __call__(self, func: _F) -> _F:
        """Identity function."""
        return func
