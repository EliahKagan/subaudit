"""Fixtures and helpers used in multiple test modules."""

__all__ = [
    'MaybeRaiser',
    'maybe_raise',
]

from typing import Callable, ClassVar, Type

import attrs
import pytest


class _FakeError(Exception):
    """Fake exception for testing."""


@attrs.frozen
class MaybeRaiser:
    """
    A callable that raises a fake error or does nothing.

    The ``maybe_raise`` fixture returns an instance of this.
    """

    Exception: ClassVar[Type[_FakeError]] = _FakeError
    """The exception type to raise (when raising at all)."""

    raises: bool
    """Whether to raise an exception or not, when called."""

    def __call__(self) -> None:
        """Maybe raise an exception."""
        if self.raises:
            raise self.Exception


@pytest.fixture(params=[False, True])
def maybe_raise(request: pytest.FixtureRequest) -> Callable[[], None]:
    """
    An object that, when called, either raises ``_FakeError`` or do nothing.

    This fixture multiplies tests, covering raising and non-raising cases.
    """
    return MaybeRaiser(request.param)
