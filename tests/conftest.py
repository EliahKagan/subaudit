"""Fixtures and helpers used in multiple test modules."""

__all__ = [
    'MaybeRaiser',
    'maybe_raise',
    'AnyHook',
    'any_hook',
]

from typing import Callable, ClassVar, ContextManager, List, Type, TypeVar

import attrs
import pytest
from typing_extensions import Protocol

import subaudit

_R = TypeVar('_R')
"""Function-level output type variable."""


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


class AnyHook(Protocol):
    """Protocol for the ``Hook`` interface."""

    # pylint: disable=missing-function-docstring  # This is a protocol.

    __slots__ = ()

    def subscribe(self, event: str, listener: Callable[..., None]) -> None: ...

    def unsubscribe(self, event: str, listener: Callable[..., None]) -> None:
        ...

    def listening(
        self, event: str, listener: Callable[..., None],
    ) -> ContextManager[None]: ...

    def extracting(
        self, event: str, extractor: Callable[..., _R],
    ) -> ContextManager[List[_R]]: ...


class _TopLevel:
    """
    Test double providing top-level functions from the ``subaudit`` module.

    This is so the tests of the top-level functions don't depend on them being
    instance methods (of ``Hook``), which may change. (They may delegate to
    instance methods in the future. We would lose ``__self__``, but they could
    have their own docstrings.) Otherwise, we would just access ``__self__`` on
    one of them and use that ``Hook``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Python code representation."""
        return f'{type(self).__name__}()'

    def subscribe(self, event: str, listener: Callable[..., None]) -> None:
        """Call the top-level ``subscribe``."""
        return subaudit.subscribe(event, listener)

    def unsubscribe(self, event: str, listener: Callable[..., None]) -> None:
        """Call the top-level ``unsubscribe``."""
        return subaudit.unsubscribe(event, listener)

    def listening(
        self, event: str, listener: Callable[..., None],
    ) -> ContextManager[None]:
        """Call the top-level ``listening``."""
        return subaudit.listening(event, listener)

    def extracting(
        self, event: str, extractor: Callable[..., _R],
    ) -> ContextManager[List[_R]]:
        """Call the top-level ``extracting``."""
        return subaudit.extracting(event, extractor)


@pytest.fixture(params=[subaudit.Hook, _TopLevel])
def any_hook(request: pytest.FixtureRequest) -> AnyHook:
    """
    ``Hook`` instance or wrapper for the top-level functions (pytest fixture).

    This multiplies tests, covering new instances and module-level functions.
    """
    return request.param()
