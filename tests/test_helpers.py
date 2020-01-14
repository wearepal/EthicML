"""Test helper functions"""
import pytest

from ethicml.common import implements


def test_implements() -> None:
    """test the implements decorator"""

    class BaseClass:
        def func(self) -> None:
            """Do nothing."""

        def no_docstring(self) -> None:
            pass

    class CorrectImplementation(BaseClass):
        @implements(BaseClass)
        def func(self) -> None:
            pass

    with pytest.raises(AssertionError):

        class IncorrectImplementation(BaseClass):
            @implements(BaseClass)
            def wrong_func(self) -> None:
                pass

    with pytest.raises(AssertionError):

        class NoDocstringImpl(BaseClass):
            @implements(BaseClass)
            def no_docstring(self) -> None:
                pass
