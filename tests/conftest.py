"""Fixtures and helpers used in multiple test modules."""

__all__ = [
    'MaybeRaiser',
    'maybe_raise',
    'AnyHook',
    'any_hook',
    'MultiSupplier',
    'hook',
    'make_hooks',
]

from typing import (
    Callable,
    ClassVar,
    ContextManager,
    Generic,
    List,
    Tuple,
    Type,
    TypeVar,
)

import attrs
import pytest
from typing_extensions import Protocol

import subaudit

_R = TypeVar('_R')
"""Function-level output type variable."""

_R_co = TypeVar('_R_co', covariant=True)
"""Class-level output type variable (covariant)."""


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


class MultiSupplier(Generic[_R_co]):
    """Adapter of a single-item supplier to produce multiple items."""

    __slots__ = ('_supplier',)

    _supplier: Callable[[], _R_co]

    def __init__(self, supplier: Callable[[], _R_co]) -> None:
        """Create a maker from the given single-item supplier."""
        self._supplier = supplier

    def __repr__(self) -> str:
        """Vaguely code-like representation for debugging."""
        return f'{type(self).__name__}({self._supplier!r})'

    def __call__(self, count: int) -> Tuple[_R_co, ...]:
        """Make the specific number (``count``) of things."""
        return tuple(self._supplier() for _ in range(count))


def _make_hook() -> subaudit.Hook:
    """Create a ``Hook`` instance."""
    return subaudit.Hook()


@pytest.fixture
def hook() -> subaudit.Hook:
    """``Hook`` instance (pytest fixture)."""
    return _make_hook()


@pytest.fixture
def make_hooks() -> MultiSupplier[subaudit.Hook]:
    """Supplier of multiple ``Hook`` instances (pytest fixture)."""
    return MultiSupplier(_make_hook)
