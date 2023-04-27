"""Fixtures and helpers used in multiple test modules."""

__all__ = [
    'MaybeRaiser',
    'maybe_raise',
    'AnyHook',
    'any_hook',
    'MultiSupplier',
    'hook',
    'make_hooks',
    'DerivedHookFixture',
    'derived_hook',
]

import functools
from typing import (
    Any,
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
import mock
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


# FIXME: Decide if this really should be defined here. If listening and
#        extracting are tested in the same module as each other, I think this
#        and its supporting classes can be moved to that module.
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


class _UnboundMethodMock(mock.Mock):
    """A ``Mock`` that is also a descriptor, to behave like a function."""

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        """When accessed through an instance, produce a "bound method"."""
        return self if instance is None else functools.partial(self, instance)


@attrs.frozen
class DerivedHookFixture:
    """
    A newly created ``Hook`` subclass's method mocks, and an instance.

    This is what the ``derived_hook`` fixture provides.
    """

    subscribe_method: mock.Mock
    """Mock of the unbound ``subscribe`` method."""

    unsubscribe_method: mock.Mock
    """Mock of the unbound ``unsubscribe`` method."""

    listening_method: mock.Mock
    """Mock of the unbound ``listening`` context manager method."""

    extracting_method: mock.Mock
    """Mock of the unbound ``extracting`` context manager method."""

    instance: subaudit.Hook
    """Instance of the ``Hook`` subclass whose methods are mocked."""


# FIXME: Decide if this really should be defined here. If listening and
#        extracting are tested in the same module as each other, then I should
#        look into removing the one dependence of a repr test on this fixture
#        (which was sort of a questionable choice anyway) and moving this and
#        supporting classes into the module of listening and extracting tests.
@pytest.fixture
def derived_hook() -> DerivedHookFixture:
    """Make a new ``Hook`` subclass with methods mocked (pytest fixture)."""
    subscribe_method = _UnboundMethodMock(wraps=subaudit.Hook.subscribe)
    unsubscribe_method = _UnboundMethodMock(wraps=subaudit.Hook.unsubscribe)
    listening_method = _UnboundMethodMock(wraps=subaudit.Hook.listening)
    extracting_method = _UnboundMethodMock(wraps=subaudit.Hook.extracting)

    class MockedSubscribeUnsubscribeHook(subaudit.Hook):
        """``Hook`` subclass with mocked methods, for a single test."""
        subscribe = subscribe_method
        unsubscribe = unsubscribe_method
        listening = listening_method
        extracting = extracting_method

    return DerivedHookFixture(
        subscribe_method=subscribe_method,
        unsubscribe_method=unsubscribe_method,
        listening_method=listening_method,
        extracting_method=extracting_method,
        instance=MockedSubscribeUnsubscribeHook(),
    )
