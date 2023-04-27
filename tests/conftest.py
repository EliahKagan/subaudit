"""Fixtures and helpers used in multiple test modules."""

__all__ = [
    'MaybeRaiser',
    'maybe_raise',
    'AnyHook',
    'MultiSupplier',
    'DerivedHookFixture',
    'MockLike',
    'MockListener',
]

import functools
from typing import (
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
import uuid

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


@pytest.fixture(name='any_hook', params=[subaudit.Hook, _TopLevel])
def any_hook_fixture(request: pytest.FixtureRequest) -> AnyHook:
    """
    ``Hook`` instance or wrapper for the top-level functions.

    This fixture multiplies tests, covering both new instances of ``Hook`` and
    the module-level functions that use an existing instance.
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


@pytest.fixture(name='hook')
def hook_fixture() -> subaudit.Hook:
    """
    ``Hook`` instance.

    This fixture provides a newly created ``Hook`` instance.
    """
    return _make_hook()


@pytest.fixture(name='make_hooks')
def make_hooks_fixture() -> MultiSupplier[subaudit.Hook]:
    """
    Supplier of multiple ``Hook`` instances.

    This fixture provides a callable object that returns a tuple of separate,
    new ``Hook`` instances, whose length is specified by the argument passed.
    """
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
@pytest.fixture(name='derived_hook')
def derived_hook_fixture() -> DerivedHookFixture:
    """
    Make a new ``Hook`` subclass with methods mocked.

    This fixture creates a new ``Hook`` subclass, separate from any other
    classes, including prior subclasses it has created elsewhere. This class's
    methods are mocked in such a way that, when they are called on an instance,
    they receive and record the ``self`` argument, as well as others. (This
    differs from the behavior of a plain ``Mock`` or ``MagicMock`` on a class,
    when called through an instance.) It also instantiates the new class.

    The methods can be accessed as the ``subscribe_method``,
    ``unsubscribe_method``, ``listening_method``, and ``extracting_method``
    attributes on the fixture object. The instance can be accessed as the
    ``instance`` attribute.
    """
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


def _make_event() -> str:
    """Create a randomly generated fake event name."""
    return f'test-subaudit-{uuid.uuid4()}'


@pytest.fixture(name='event')
def event_fixture() -> str:
    """
    Randomly generated fake event name.

    This fixture provides a string for use as an event name, incorporating a
    new randomly generated UUID. (The probability of uniqueness is very high.)
    """
    return _make_event()


@pytest.fixture(name='make_events')
def make_events_fixture() -> MultiSupplier[str]:
    """
    Supplier of multiple randomly generated fake event names.

    This fixture provides a callable object that returns a tuple of separate
    new randomly generated fake event names, whose length is specified by the
    argument passed.
    """
    return MultiSupplier(_make_event)


@pytest.fixture(name='null_listener')
def null_listener_fixture() -> Callable[..., None]:
    """
    Listener that does nothing, for spec-ing.

    This fixture is for building other fixtures. It provides a function they
    can pass as a ``spec`` argument for ``Mock``, ``MagicMock``, patchers, etc.
    """
    return lambda *_: None


class MockLike(Protocol):
    """
    Protocol for objects with ``assert_*`` methods and call spying we need.
    """

    # pylint: disable=missing-function-docstring  # This is a protocol.

    __slots__ = ()

    @property
    def call_count(self) -> int: ...

    # FIXME: Name the return type in a way that does not violate encapsulation.
    @property
    def mock_calls(self) -> mock.mock._CallList: ...

    def assert_called_once(self) -> None: ...

    def assert_called_with(self, *args: Any, **kwargs: Any) -> None: ...

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None: ...

    def assert_not_called(self) -> None: ...


class MockListener(MockLike, Protocol):
    """Protocol for a listener that supports some of the ``Mock`` interface."""

    # pylint: disable=missing-function-docstring  # This is a protocol.

    __slots__ = ()

    def __call__(self, *args: Any) -> None: ...

    @property
    def side_effect(self) -> Optional[Callable[..., Any]]: ...

    @side_effect.setter
    def side_effect(self, __value: Optional[Callable[..., Any]]) -> None: ...


@pytest.fixture(name='listener')
def listener_fixture(null_listener: Callable[..., None]) -> MockListener:
    """
    Mock listener.

    This fixture provides a mock suitable for use as a listener.
    """
    return mock.Mock(spec=null_listener)


@pytest.fixture(name='make_listeners')
def make_listeners_fixture(
    null_listener: Callable[..., None],
) -> MultiSupplier[MockListener]:
    """
    Supplier of multiple mock listeners.

    This fixture provides a callable object that returns a tuple of separate
    mocks suitable for use as listeners.
    """
    return MultiSupplier(lambda: mock.Mock(spec=null_listener))
