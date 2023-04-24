# Copyright (c) 2023 Eliah Kagan
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

"""Tests for the subaudit module."""

# TODO: Maybe split this into multiple modules.

import contextlib
import datetime
import enum
import functools
import os
import pathlib
import random
import re
import sys
import threading
from types import MethodType
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)
import unittest
# TODO: Find a way to hint like _Call and _CallList, yet respect encapsulation.
from unittest.mock import _Call, _CallList, Mock, call
import uuid

import attrs
import clock_timer
import pytest
from pytest import CaptureFixture, FixtureRequest, MonkeyPatch
from pytest_mock import MockerFixture
from pytest_subtests import SubTests
from typing_extensions import Protocol, Self

import subaudit
from subaudit import Hook

_R = TypeVar('_R')
"""Function-level output type variable."""

_R_co = TypeVar('_R_co', covariant=True)
"""Class-level output type variable (covariant)."""

_STDIN_FILENO = 0
"""File descriptor of real standard input."""


class _FakeError(Exception):
    """Fake exception for testing."""


@pytest.fixture(name='maybe_raise', params=[False, True])
def _maybe_raise_fixture(request: FixtureRequest) -> Callable[[], None]:
    """
    Function that either raises _FakeError or do nothing (pytest fixture).

    This multiplies tests, covering raising and non-raising cases.
    """
    def maybe_raise_now() -> None:
        if request.param:
            raise _FakeError

    return maybe_raise_now


class _AnyHook(Protocol):
    """Protocol for the Hook interface."""

    # pylint: disable=missing-function-docstring  # This is a protocol.

    __slots__ = ()

    def subscribe(self, event: str, listener: Callable[..., None]) -> None: ...

    def unsubscribe(self, event: str, listener: Callable[..., None]) -> None:
        ...

    def listening(
        self, event: str, listener: Callable[..., None],
    ) -> contextlib.AbstractContextManager[None]: ...

    def extracting(
        self, event: str, extractor: Callable[..., _R],
    ) -> contextlib.AbstractContextManager[List[_R]]: ...


class TopLevel:
    """
    Test double providing top-level Hook functions from the subaudit modules.

    This is so the tests of those functions don't depend on them being instance
    methods, which may change. (They may delegate to instance methods in the
    future. We would lose __self__, but they could have their own docstrings.)
    Otherwise, we would just access __self__ on one of them and use that Hook.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Python code representation."""
        return f'{type(self).__name__}()'

    def subscribe(self, event: str, listener: Callable[..., None]) -> None:
        """Call the top-level subscribe."""
        return subaudit.subscribe(event, listener)

    def unsubscribe(self, event: str, listener: Callable[..., None]) -> None:
        """Call the top-level unsubscribe."""
        return subaudit.unsubscribe(event, listener)

    def listening(
        self, event: str, listener: Callable[..., None],
    ) -> contextlib.AbstractContextManager[None]:
        """Call the top-level listening."""
        return subaudit.listening(event, listener)

    def extracting(
        self, event: str, extractor: Callable[..., _R],
    ) -> contextlib.AbstractContextManager[List[_R]]:
        """Call the top-level extracting."""
        return subaudit.extracting(event, extractor)


@pytest.fixture(name='any_hook', params=[Hook, TopLevel])
def _any_hook_fixture(request: FixtureRequest) -> _AnyHook:
    """
    Hook instance or wrapper for the top-level functions (pytest fixture).

    This multiplies tests, covering new instances and module-level functions.
    """
    return request.param()


class _MultiSupplier(Generic[_R_co]):
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
        """Make the specific number (count) of things."""
        return tuple(self._supplier() for _ in range(count))


def _make_hook() -> Hook:
    """Create a hook instance."""
    return Hook()


@pytest.fixture(name='hook')
def _hook_fixture() -> Hook:
    """Hook instance (pytest fixture)."""
    return _make_hook()


@pytest.fixture(name='make_hooks')
def _make_hooks_fixture() -> _MultiSupplier[Hook]:
    """Supplier of multiple Hook instances (pytest fixture)."""
    return _MultiSupplier(_make_hook)


class _UnboundMethodMock(Mock):
    """A mock that is also a descriptor, to behave like a function."""

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        """When accessed through an instance, produce a "bound method"."""
        return self if instance is None else functools.partial(self, instance)


@attrs.frozen
class _DerivedHookFixture:
    """
    New Hook subclass method mocks and instance, for the derived_hook fixture.
    """

    # pylint: disable=too-few-public-methods  # This is an attrs data class.

    subscribe_method: Mock
    """Mock of the unbound subscribe method."""

    unsubscribe_method: Mock
    """Mock of the unbound unsubscribe method."""

    listening_method: Mock
    """Mock of the unbound listening context manager method."""

    extracting_method: Mock
    """Mock of the unbound extracting context manager method."""

    instance: Hook
    """Instance of the Hook subclass whose methods are mocked."""


@pytest.fixture(name='derived_hook')
def _derived_hook_fixture() -> _DerivedHookFixture:
    """Make a new Hook subclass with methods mocked (pytest fixture)."""
    subscribe_method = _UnboundMethodMock(wraps=Hook.subscribe)
    unsubscribe_method = _UnboundMethodMock(wraps=Hook.unsubscribe)
    listening_method = _UnboundMethodMock(wraps=Hook.listening)
    extracting_method = _UnboundMethodMock(wraps=Hook.extracting)

    class MockedSubscribeUnsubscribeHook(Hook):
        """Hook subclass with mocked methods, for a single test."""
        subscribe = subscribe_method
        unsubscribe = unsubscribe_method
        listening = listening_method
        extracting = extracting_method

    return _DerivedHookFixture(
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
def _event_fixture() -> str:
    """Randomly generated fake event name (pytest fixture)."""
    return _make_event()


@pytest.fixture(name='make_events')
def _make_events_fixture() -> _MultiSupplier[str]:
    """
    Supplier of multiple randomly generated fake event names (pytest fixture).
    """
    return _MultiSupplier(_make_event)


class _MockLike(Protocol):  # TODO: Drop any members that aren't needed.
    """Protocol for objects with assert_* methods and call spying we need."""

    # pylint: disable=missing-function-docstring  # This is a protocol.

    __slots__ = ()

    @property
    def called(self) -> bool: ...

    @property
    def call_count(self) -> int: ...

    @property
    def mock_calls(self) -> _CallList: ...

    def assert_called(self) -> None: ...

    def assert_called_once(self) -> None: ...

    def assert_called_with(self, *args: Any, **kwargs: Any) -> None: ...

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None: ...

    def assert_any_call(self, *args: Any, **kwargs: Any) -> None: ...

    def assert_has_calls(
        self, calls: Sequence[_Call], any_order: bool = False,
    ) -> None: ...

    def assert_not_called(self) -> None: ...


class _MockListener(_MockLike, Protocol):
    """Protocol for a listener that supports some of the Mock interface."""

    # pylint: disable=missing-function-docstring  # This is a protocol.

    __slots__ = ()

    def __call__(self, *args: Any) -> None: ...

    @property
    def side_effect(self) -> Optional[Callable[..., Any]]: ...

    @side_effect.setter
    def side_effect(self, __value: Optional[Callable[..., Any]]) -> None: ...


def _null_listener(*_: Any) -> None:
    """Listener that does nothing, for spec-ing."""


def _make_listener() -> _MockListener:
    """Create a mock listener."""
    return Mock(spec=_null_listener)


@pytest.fixture(name='listener')
def _listener_fixture() -> _MockListener:
    """Mock listener (pytest fixture)."""
    return _make_listener()


@pytest.fixture(name='make_listeners')
def _make_listeners_fixture() -> _MultiSupplier[_MockListener]:
    """Supplier of multiple mock listeners (pytest fixture)."""
    return _MultiSupplier(_make_listener)


@pytest.fixture(name='equal_listeners', params=[2, 3, 5])
def _equal_listeners_fixture(
    request: FixtureRequest,
) -> Tuple[_MockListener, ...]:
    """Listeners that are different objects but all equal (pytest fixture)."""
    group_key = object()

    def in_group(other: object) -> bool:
        return getattr(other, 'group_key', None) is group_key

    def make_mock() -> _MockListener:
        return Mock(
            spec=_null_listener,
            __eq__=Mock(side_effect=in_group),
            __hash__=Mock(return_value=hash(group_key)),
            group_key=group_key,
        )

    return tuple(make_mock() for _ in range(request.param))


@attrs.frozen
class _Extract:
    """
    Auditing event arguments extracted by a custom extractor.

    The point of this type is to be separate from anything in, or used by, the
    code under test, so no excessively specific behavior wrongly passes tests
    of more general behavior.
    """

    # pylint: disable=too-few-public-methods  # This is an attrs data class.

    args: Tuple[Any, ...]
    """Event arguments extracted in a test."""

    @classmethod
    def from_separate_args(cls, *args: Any) -> Self:
        """Create an _Extract instance from separately passed arguments."""
        return cls(args=args)


class _MockExtractor(_MockLike, Protocol):
    """Protocol for an extractor that supports some of the Mock interface."""

    __slots__ = ()

    def __call__(self, *args: Any) -> _Extract: ...


@pytest.fixture(name='extractor')
def _extractor_fixture() -> _MockExtractor:
    """Mock extractor (pytest fixture). Returns a tuple of its arguments."""
    return Mock(wraps=_Extract.from_separate_args)


@attrs.frozen
class _ReprAsserter:
    """Callable to assert correct Hook repr."""

    # pylint: disable=too-few-public-methods  # Class for clearer type hinting.

    def __call__(
        self,
        hook: Hook,
        summary_pattern: str,
        *,
        type_name_pattern: str = 'Hook',
    ) -> None:
        """Assert the repr is the non-code style with a matching summary."""
        regex = rf'<{type_name_pattern} at 0x[0-9a-fA-F]+: {summary_pattern}>'
        assert re.fullmatch(regex, repr(hook))


@pytest.fixture(name='assert_repr_summary')
def _assert_repr_summary_fixture() -> _ReprAsserter:
    """
    Function to assert a hook has a correct repr (pytest fixture).

    The repr is asserted to be in the non-code style, containing general
    information followed by a summary matching the specific pattern passed.
    """
    return _ReprAsserter()


@attrs.frozen
class _MockLockFixture:
    """A mock lock factory and Hook that uses it, for the mock_lock fixture."""

    # pylint: disable=too-few-public-methods  # This is an attrs data class.

    lock_factory: Mock
    """The mock lock (mutex). This does not do any real locking."""

    hook: Hook
    """The Hook instance that was created using the mock lock."""


@enum.unique
class Scope(enum.Enum):
    """Ways to supply a mock lock to a Hook: pass locally or patch globally."""

    LOCAL = enum.auto()
    """Pass locally."""

    GLOBAL = enum.auto()
    """Patch globally."""


@pytest.fixture(name='mock_lock', params=[Scope.LOCAL, Scope.GLOBAL])
def _mock_lock_fixture(
    request: FixtureRequest, mocker: MockerFixture,
) -> Generator[_MockLockFixture, None, None]:
    """A Hook created with its lock mocked (pytest fixture)."""
    lock_factory = mocker.MagicMock(threading.Lock)

    if request.param is Scope.LOCAL:
        hook = Hook(sub_lock_factory=lock_factory)
    elif request.param is Scope.GLOBAL:
        mocker.patch('threading.Lock', lock_factory)
        hook = Hook()
    else:
        # We've exhausted the enumeration, so the "scope" is the wrong type.
        raise TypeError('scope must be a Scope (one of LOCAL or GLOBAL)')

    yield _MockLockFixture(lock_factory=lock_factory, hook=hook)


# pylint: disable=missing-function-docstring  # Tests are descriptively named.


@pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason='Python 3.8+ has sys.audit.',
    raises=AttributeError,
    strict=True,
)
def test_audit_is_sys_audit_since_3_8() -> None:
    assert subaudit.audit is sys.audit


@pytest.mark.xfail(
    sys.version_info >= (3, 8),
    reason='Python 3.8+ has sys.audit.',
    raises=ImportError,
    strict=True,
)
def test_audit_is_sysaudit_audit_before_3_8() -> None:
    # pylint: disable=import-error,import-outside-toplevel
    import sysaudit  # type: ignore[import]
    assert subaudit.audit is sysaudit.audit


@pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason='Python 3.8+ has sys.addaudithook.',
    raises=AttributeError,
    strict=True,
)
def test_addaudithook_is_sys_addaudithook_since_3_8() -> None:
    assert subaudit.addaudithook is sys.addaudithook


@pytest.mark.xfail(
    sys.version_info >= (3, 8),
    reason='Python 3.8+ has sys.addaudithook.',
    raises=ImportError,
    strict=True,
)
def test_addaudithook_is_sysaudit_addaudithook_before_3_8() -> None:
    # pylint: disable=import-error,import-outside-toplevel
    import sysaudit  # type: ignore[import]
    assert subaudit.addaudithook is sysaudit.addaudithook


def test_subscribed_listener_observes_event(
    any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    any_hook.subscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_called_once_with('a', 'b', 'c')


def test_unsubscribed_listener_does_not_observe_event(
    any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    any_hook.subscribe(event, listener)
    any_hook.unsubscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_not_called()


def test_subscribed_listener_does_not_observe_other_event(
    any_hook: _AnyHook,
    make_events: _MultiSupplier[str],
    listener: _MockListener,
) -> None:
    """Subscribing to one event doesn't observe other events."""
    event1, event2 = make_events(2)
    any_hook.subscribe(event1, listener)
    subaudit.audit(event2, 'a', 'b', 'c')
    listener.assert_not_called()


def test_listener_can_subscribe_multiple_events(
    any_hook: _AnyHook,
    make_events: _MultiSupplier[str],
    listener: _MockListener,
) -> None:
    event1, event2 = make_events(2)
    any_hook.subscribe(event1, listener)
    any_hook.subscribe(event2, listener)
    subaudit.audit(event1, 'a', 'b', 'c')
    subaudit.audit(event2, 'd', 'e')
    assert listener.mock_calls == [call('a', 'b', 'c'), call('d', 'e')]


def test_listeners_called_in_subscribe_order(
    any_hook: _AnyHook,
    event: str,
    make_listeners: _MultiSupplier[_MockListener],
) -> None:
    ordering: List[int] = []
    listener1, listener2, listener3 = make_listeners(3)
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)
    listener3.side_effect = functools.partial(ordering.append, 3)

    any_hook.subscribe(event, listener1)
    any_hook.subscribe(event, listener2)
    any_hook.subscribe(event, listener3)
    subaudit.audit(event)

    assert ordering == [1, 2, 3]


def test_listeners_called_in_subscribe_order_after_others_unsubscribe(
    any_hook: _AnyHook,
    event: str,
    make_listeners: _MultiSupplier[_MockListener],
) -> None:
    ordering: List[int] = []
    listener1, listener2, listener3, listener4 = make_listeners(4)
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)
    listener3.side_effect = functools.partial(ordering.append, 3)
    listener4.side_effect = functools.partial(ordering.append, 4)

    any_hook.subscribe(event, listener1)
    any_hook.subscribe(event, listener2)
    any_hook.subscribe(event, listener3)
    any_hook.subscribe(event, listener4)
    any_hook.unsubscribe(event, listener1)
    any_hook.unsubscribe(event, listener3)
    subaudit.audit(event)

    assert ordering == [2, 4]


def test_listeners_called_in_new_order_after_resubscribe(
    any_hook: _AnyHook,
    event: str,
    make_listeners: _MultiSupplier[_MockListener],
) -> None:
    ordering: List[int] = []
    listener1, listener2 = make_listeners(2)
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)

    any_hook.subscribe(event, listener1)
    any_hook.subscribe(event, listener2)
    any_hook.unsubscribe(event, listener1)
    any_hook.subscribe(event, listener1)
    subaudit.audit(event)

    assert ordering == [2, 1]


def test_cannot_unsubscribe_if_never_subscribed(
    any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    with pytest.raises(ValueError):
        any_hook.unsubscribe(event, listener)


def test_cannot_unsubscribe_if_no_longer_subscribed(
    any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    any_hook.subscribe(event, listener)
    any_hook.unsubscribe(event, listener)
    with pytest.raises(ValueError):
        any_hook.unsubscribe(event, listener)


@pytest.mark.parametrize('count', [0, 2, 3, 10])
def test_listener_observes_event_as_many_times_as_subscribed(
    count: int, any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    for _ in range(count):
        any_hook.subscribe(event, listener)
    subaudit.audit(event)
    assert listener.call_count == count


@pytest.mark.parametrize('count', [2, 3, 10])
def test_can_unsubscribe_as_many_times_as_subscribed(
    count: int, any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    for _ in range(count):
        any_hook.subscribe(event, listener)
    try:
        for _ in range(count):
            any_hook.subscribe(event, listener)
    except ValueError as error:
        pytest.fail(
            f"Couldn't subscribe then unsubscribe {count} times: {error!r}")


@pytest.mark.parametrize('count', [2, 3, 10])
def test_cannot_unsubscribe_more_times_than_subscribed(
    count: int, any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    for _ in range(count):
        any_hook.subscribe(event, listener)
    with pytest.raises(ValueError):
        for _ in range(count + 1):
            any_hook.unsubscribe(event, listener)


def test_unsubscribe_keeps_other_listener(
    any_hook: _AnyHook,
    event: str,
    make_listeners: _MultiSupplier[_MockListener],
) -> None:
    """Unsubscribing one listener doesn't prevent another from observing."""
    listener1, listener2 = make_listeners(2)
    any_hook.subscribe(event, listener1)
    any_hook.subscribe(event, listener2)
    any_hook.unsubscribe(event, listener1)
    subaudit.audit(event, 'a', 'b', 'c')
    listener2.assert_called_once_with('a', 'b', 'c')


def test_unsubscribe_removes_last_equal_listener(
    any_hook: _AnyHook, event: str, equal_listeners: Tuple[_MockListener, ...],
) -> None:
    for listener in equal_listeners:
        any_hook.subscribe(event, listener)
    any_hook.unsubscribe(event, equal_listeners[0])
    subaudit.audit(event, 'a', 'b', 'c')
    equal_listeners[-1].assert_not_called()


def test_unsubscribe_keeps_non_last_equal_listeners(
    subtests: SubTests,
    any_hook: _AnyHook,
    event: str,
    equal_listeners: Tuple[_MockListener, ...],
) -> None:
    """Unsubscribing removes no equal listeners besides the last subscribed."""
    for listener in equal_listeners:
        any_hook.subscribe(event, listener)
    any_hook.unsubscribe(event, equal_listeners[0])
    subaudit.audit(event, 'a', 'b', 'c')

    for index, listener in enumerate(equal_listeners[:-1]):
        with subtests.test(listener_index=index):
            listener.assert_called_once_with('a', 'b', 'c')


def test_cannot_unsubscribe_listener_from_other_hook(
    make_hooks: _MultiSupplier[Hook], event: str, listener: _MockListener,
) -> None:
    hook1, hook2 = make_hooks(2)
    hook1.subscribe(event, listener)
    with pytest.raises(ValueError):
        hook2.unsubscribe(event, listener)


def test_instance_construction_does_not_add_audit_hook(
    mocker: MockerFixture,
) -> None:
    """Hook is lazy, not adding an audit hook before a listener subscribes."""
    mock = mocker.patch('subaudit.addaudithook')
    Hook()
    mock.assert_not_called()


def test_instance_adds_audit_hook_on_first_subscribe(
    mocker: MockerFixture, hook: Hook, event: str, listener: _MockListener,
) -> None:
    mock = mocker.patch('subaudit.addaudithook')
    hook.subscribe(event, listener)
    mock.assert_called_once()


def test_instance_does_not_add_audit_hook_on_second_subscribe(
    mocker: MockerFixture,
    hook: Hook,
    make_events: _MultiSupplier[str],
    make_listeners: _MultiSupplier[_MockListener],
) -> None:
    event1, event2 = make_events(2)
    listener1, listener2 = make_listeners(2)
    hook.subscribe(event1, listener1)
    mock = mocker.patch('subaudit.addaudithook')
    hook.subscribe(event2, listener2)
    mock.assert_not_called()


def test_second_instance_adds_audit_hook_on_first_subscribe(
    mocker: MockerFixture,
    make_hooks: _MultiSupplier[Hook],
    make_events: _MultiSupplier[str],
    make_listeners: _MultiSupplier[_MockListener],
) -> None:
    """Different Hook objects do not share the same audit hook."""
    hook1, hook2 = make_hooks(2)
    event1, event2 = make_events(2)
    listener1, listener2 = make_listeners(2)
    hook1.subscribe(event1, listener1)
    mock = mocker.patch('subaudit.addaudithook')
    hook2.subscribe(event2, listener2)
    mock.assert_called_once()


def test_listening_does_not_observe_before_enter(
    any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    """The call to listening does not itself subscribe."""
    context_manager = any_hook.listening(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager:
        pass
    listener.assert_not_called()


def test_listening_does_not_call_subscribe_before_enter(
    derived_hook: _DerivedHookFixture, event: str, listener: _MockListener,
) -> None:
    derived_hook.instance.listening(event, listener)
    derived_hook.subscribe_method.assert_not_called()


def test_listening_observes_between_enter_and_exit(
    any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    """In the block of a with statement, the listener is subscribed."""
    with any_hook.listening(event, listener):
        subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_called_once_with('a', 'b', 'c')


def test_listening_enter_calls_subscribe(
    derived_hook: _DerivedHookFixture, event: str, listener: _MockListener,
) -> None:
    """An overridden subscribe method will be used by listening."""
    with derived_hook.instance.listening(event, listener):
        derived_hook.subscribe_method.assert_called_once_with(
            derived_hook.instance,
            event,
            listener,
        )


def test_listening_does_not_observe_after_exit(
    maybe_raise: Callable[[], None],
    any_hook: _AnyHook,
    event: str,
    listener: _MockListener,
) -> None:
    """After exiting the with statement, the listener is not subscribed."""
    with contextlib.suppress(_FakeError):
        with any_hook.listening(event, listener):
            maybe_raise()
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_not_called()


def test_listening_exit_calls_unsubscribe(
    subtests: SubTests,
    maybe_raise: Callable[[], None],
    derived_hook: _DerivedHookFixture,
    event: str,
    listener: _MockListener,
) -> None:
    """An overridden unsubscribe method will be called by listening."""
    with contextlib.suppress(_FakeError):
        with derived_hook.instance.listening(event, listener):
            with subtests.test(where='inside-with-block'):
                derived_hook.unsubscribe_method.assert_not_called()
            maybe_raise()

    with subtests.test(where='after-with-block'):
        derived_hook.unsubscribe_method.assert_called_once_with(
            derived_hook.instance,
            event,
            listener,
        )


def test_listening_observes_only_between_enter_and_exit(
    maybe_raise: Callable[[], None],
    any_hook: _AnyHook,
    event: str,
    listener: _MockListener,
) -> None:
    """The listening context manager works in (simple yet) nontrivial usage."""
    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with any_hook.listening(event, listener):
            subaudit.audit(event, 'd')
            subaudit.audit(event, 'e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert listener.mock_calls == [call('d'), call('e', 'f')]


def test_listening_enter_returns_none(
    any_hook: _AnyHook, event: str, listener: _MockListener,
) -> None:
    """The listening context manager isn't meant to be used with "as"."""
    with any_hook.listening(event, listener) as context:
        pass
    assert context is None


def test_extracting_does_not_observe_before_enter(
    any_hook: _AnyHook, event: str, extractor: _MockExtractor,
) -> None:
    context_manager = any_hook.extracting(event, extractor)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager:
        pass
    extractor.assert_not_called()


def test_extracting_does_not_extract_before_enter(
    any_hook: _AnyHook, event: str, extractor: _MockExtractor,
) -> None:
    context_manager = any_hook.extracting(event, extractor)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager as extracts:
        pass
    assert extracts == []


def test_extracting_does_not_call_subscribe_before_enter(
    derived_hook: _DerivedHookFixture, event: str, extractor: _MockExtractor,
) -> None:
    derived_hook.instance.extracting(event, extractor)
    derived_hook.subscribe_method.assert_not_called()


def test_extracting_observes_between_enter_and_exit(
    any_hook: _AnyHook, event: str, extractor: _MockExtractor,
) -> None:
    with any_hook.extracting(event, extractor):
        subaudit.audit(event, 'a', 'b', 'c')
    extractor.assert_called_once_with('a', 'b', 'c')


def test_extracting_extracts_between_enter_and_exit(
    any_hook: _AnyHook, event: str, extractor: _MockExtractor,
) -> None:
    with any_hook.extracting(event, extractor) as extracts:
        subaudit.audit(event, 'a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_enter_calls_subscribe_exactly_once(
    derived_hook: _DerivedHookFixture, event: str, extractor: _MockExtractor,
) -> None:
    with derived_hook.instance.extracting(event, extractor):
        derived_hook.subscribe_method.assert_called_once()


def test_extracting_enter_passes_subscribe_same_event_and_hook(
    subtests: SubTests,
    derived_hook: _DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with derived_hook.instance.extracting(event, extractor):
        subscribe_hook, subscribe_event, _ = (
            derived_hook.subscribe_method.mock_calls[0].args
        )
        with subtests.test(argument_name='hook'):
            assert subscribe_hook is derived_hook.instance
        with subtests.test(argument_name='event'):
            assert subscribe_event == event


def test_extracting_enter_passes_appender_to_subscribe(
    derived_hook: _DerivedHookFixture, event: str, extractor: _MockExtractor,
) -> None:
    with derived_hook.instance.extracting(event, extractor) as extracts:
        subscribe_call, = derived_hook.subscribe_method.mock_calls
        _, _, subscribe_listener = subscribe_call.args

    subscribe_listener('a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_does_not_observe_after_exit(
    maybe_raise: Callable[[], None],
    any_hook: _AnyHook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with contextlib.suppress(_FakeError):
        with any_hook.extracting(event, extractor):
            maybe_raise()
    subaudit.audit(event, 'a', 'b', 'c')
    extractor.assert_not_called()


def test_extracting_does_not_extract_after_exit(
    maybe_raise: Callable[[], None],
    any_hook: _AnyHook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    extracts: Optional[List[_Extract]] = None

    with contextlib.suppress(_FakeError):
        with any_hook.extracting(event, extractor) as extracts:
            maybe_raise()

    subaudit.audit(event, 'a', 'b', 'c')
    assert extracts == []


def test_extracting_exit_calls_unsubscribe_exactly_once(
    subtests: SubTests,
    maybe_raise: Callable[[], None],
    derived_hook: _DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with contextlib.suppress(_FakeError):
        with derived_hook.instance.extracting(event, extractor):
            with subtests.test(where='inside-with-block'):
                derived_hook.unsubscribe_method.assert_not_called()
            maybe_raise()

    with subtests.test(where='after-with-block'):
        derived_hook.unsubscribe_method.assert_called_once()


def test_extracting_exit_passes_unsubscribe_same_event_and_hook(
    subtests: SubTests,
    maybe_raise: Callable[[], None],
    derived_hook: _DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with contextlib.suppress(_FakeError):
        with derived_hook.instance.extracting(event, extractor):
            maybe_raise()

    unsubscribe_hook, unsubscribe_event, _ = (
        derived_hook.unsubscribe_method.mock_calls[0].args
    )
    with subtests.test(argument_name='hook'):
        assert unsubscribe_hook is derived_hook.instance
    with subtests.test(argument_name='event'):
        assert unsubscribe_event == event


def test_extracting_exit_passes_appender_to_unsubscribe(
    maybe_raise: Callable[[], None],
    derived_hook: _DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    extracts: Optional[List[_Extract]] = None

    with contextlib.suppress(_FakeError):
        with derived_hook.instance.extracting(event, extractor) as extracts:
            maybe_raise()

    unsubscribe_call, = derived_hook.unsubscribe_method.mock_calls
    _, _, unsubscribe_listener = unsubscribe_call.args
    unsubscribe_listener('a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_observes_only_between_enter_and_exit(
    maybe_raise: Callable[[], None],
    any_hook: _AnyHook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with any_hook.extracting(event, extractor):
            subaudit.audit(event, 'd')
            subaudit.audit(event, 'e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert extractor.mock_calls == [call('d'), call('e', 'f')]


def test_extracting_extracts_only_between_enter_and_exit(
    maybe_raise: Callable[[], None],
    any_hook: _AnyHook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    extracts: Optional[List[_Extract]] = None

    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with any_hook.extracting(event, extractor) as extracts:
            subaudit.audit(event, 'd')
            subaudit.audit(event, 'e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert extracts == [_Extract(args=('d',)), _Extract(args=('e', 'f'))]


def test_extracting_subscribes_and_unsubscribes_same(
    maybe_raise: Callable[[], None],
    derived_hook: _DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with contextlib.suppress(_FakeError):
        with derived_hook.instance.extracting(event, extractor):
            maybe_raise()

    subscribe_calls = derived_hook.subscribe_method.mock_calls
    unsubscribe_calls = derived_hook.unsubscribe_method.mock_calls
    assert subscribe_calls == unsubscribe_calls


def test_extracting_delegates_to_listening(
    subtests: SubTests,
    derived_hook: _DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    """Overriding listening customizes the behavior of extracting too."""
    with contextlib.suppress(_FakeError):
        with derived_hook.instance.extracting(event, extractor):
            with subtests.test('listening called'):
                derived_hook.listening_method.assert_called_once()
            with subtests.test('listening call matches subscribe call'):
                listening_calls = derived_hook.listening_method.mock_calls
                subscribe_calls = derived_hook.subscribe_method.mock_calls
                assert listening_calls == subscribe_calls


def test_repr_shows_hook_not_installed_on_creation(
    hook: Hook, assert_repr_summary: _ReprAsserter,
) -> None:
    """A new Hook's repr has general info and the not-installed summary."""
    assert_repr_summary(hook, r'audit hook not installed')


def test_repr_shows_one_event_while_listening(
    hook: Hook,
    event: str,
    listener: _MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    with hook.listening(event, listener):
        assert_repr_summary(hook, r'watching 1 event')


def test_repr_shows_no_events_after_done_listening(
    hook: Hook,
    event: str,
    listener: _MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    with hook.listening(event, listener):
        pass
    assert_repr_summary(hook, r'watching 0 events')


def test_repr_shows_both_events_when_nested_listening(
    hook: Hook,
    make_events: _MultiSupplier[str],
    listener: _MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    event1, event2 = make_events(2)
    with hook.listening(event1, listener):
        with hook.listening(event2, listener):
            assert_repr_summary(hook, r'watching 2 events')


def test_repr_shows_one_event_after_done_listening_to_second(
    hook: Hook,
    make_events: _MultiSupplier[str],
    listener: _MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    event1, event2 = make_events(2)
    with hook.listening(event1, listener):
        with hook.listening(event2, listener):
            pass
        assert_repr_summary(hook, r'watching 1 event')


def test_repr_shows_no_events_after_done_nested_listening(
    hook: Hook,
    make_events: _MultiSupplier[str],
    listener: _MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    event1, event2 = make_events(2)
    with hook.listening(event1, listener):
        with hook.listening(event2, listener):
            pass
    assert_repr_summary(hook, r'watching 0 events')


def test_repr_shows_one_event_with_multiple_listeners_as_one(
    hook: Hook,
    event: str,
    make_listeners: _MultiSupplier[_MockListener],
    assert_repr_summary: _ReprAsserter,
) -> None:
    listener1, listener2 = make_listeners(2)
    with hook.listening(event, listener1):
        with hook.listening(event, listener2):
            assert_repr_summary(hook, r'watching 1 event')


def test_repr_uses_derived_class_type_name(
    derived_hook: _DerivedHookFixture, assert_repr_summary: _ReprAsserter,
) -> None:
    assert_repr_summary(
        derived_hook.instance,
        r'audit hook not installed',
        type_name_pattern=r'MockedSubscribeUnsubscribeHook',
    )


def test_lock_constructed_immediately(mock_lock: _MockLockFixture) -> None:
    """When a Hook is constructed, it calls its lock factory."""
    mock_lock.lock_factory.assert_called_once_with()


def test_lock_not_entered_immediately(mock_lock: _MockLockFixture) -> None:
    """When a Hook is constructed, it does not actually enter the lock."""
    mock_lock.lock_factory().__enter__.assert_not_called()


def test_subscribe_enters_and_exits_lock(
    mock_lock: _MockLockFixture, event: str, listener: _MockListener,
) -> None:
    mock_lock.hook.subscribe(event, listener)
    calls = mock_lock.lock_factory().mock_calls

    # pylint: disable=unnecessary-dunder-call  # Not really entering/exiting.
    assert calls == [call.__enter__(), call.__exit__(None, None, None)]


def test_unsubscribe_enters_and_exits_lock(
    mock_lock: _MockLockFixture, event: str, listener: _MockListener,
) -> None:
    mock_lock.hook.subscribe(event, listener)  # So that we can unsubscribe it.
    mock_lock.lock_factory().reset_mock()  # To only see calls via unsubscribe.
    mock_lock.hook.unsubscribe(event, listener)
    calls = mock_lock.lock_factory().mock_calls

    # pylint: disable=unnecessary-dunder-call  # Not really entering/exiting.
    assert calls == [call.__enter__(), call.__exit__(None, None, None)]


def test_subscribe_never_calls_acquire(
    mock_lock: _MockLockFixture, event: str, listener: _MockListener,
) -> None:
    """The lock context manager is not required to have an acquire method."""
    mock_lock.hook.subscribe(event, listener)
    mock_lock.lock_factory().acquire.assert_not_called()


def test_subscribe_never_calls_release(
    mock_lock: _MockLockFixture, event: str, listener: _MockListener,
) -> None:
    """The lock context manager is not required to have a release method."""
    mock_lock.hook.subscribe(event, listener)
    mock_lock.lock_factory().release.assert_not_called()


def test_unsubscribe_never_calls_acquire(
    mock_lock: _MockLockFixture, event: str, listener: _MockListener,
) -> None:
    """The lock context manager is not required to have an acquire method."""
    mock_lock.hook.subscribe(event, listener)  # So that we can unsubscribe it.
    mock_lock.lock_factory().reset_mock()  # To only see calls via unsubscribe.
    mock_lock.hook.unsubscribe(event, listener)
    mock_lock.lock_factory().acquire.assert_not_called()


def test_unsubscribe_never_calls_release(
    mock_lock: _MockLockFixture, event: str, listener: _MockListener,
) -> None:
    """The lock context manager is not required to have a release method."""
    mock_lock.hook.subscribe(event, listener)  # So that we can unsubscribe it.
    mock_lock.lock_factory().reset_mock()  # To only see calls via unsubscribe.
    mock_lock.hook.unsubscribe(event, listener)
    mock_lock.lock_factory().release.assert_not_called()


@pytest.mark.parametrize('cm_factory', [
    contextlib.nullcontext,
    threading.Lock,
    threading.RLock,
])
def test_lock_accepts_common_cm(
    cm_factory: subaudit.ContextManagerFactory,
    maybe_raise: Callable[[], None],
    event: str,
    listener: _MockListener,
) -> None:
    """
    A hook with a lock or nullcontext as its lock factory works for listening.

    This is test_listening_observes_only_between_enter_and_exit, but creating
    the Hook with contextlib.nullcontext, Lock, or RLock as its lock factory
    for subscribing and unsubscribing, as a simple but nontrivial check.
    """
    hook = Hook(sub_lock_factory=cm_factory)

    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with hook.listening(event, listener):
            subaudit.audit(event, 'd')
            subaudit.audit(event, 'e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert listener.mock_calls == [call('d'), call('e', 'f')]


def test_sub_lock_factory_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        # pylint: disable=too-many-function-args  # We are testing that error.
        Hook(threading.Lock)  # type: ignore[misc]


@pytest.mark.unstable
def test_top_level_functions_are_bound_methods(subtests: SubTests) -> None:
    """The module-level functions are bound methods of a Hook object."""
    top_level_functions = cast(Sequence[MethodType], [
        subaudit.subscribe,
        subaudit.unsubscribe,
        subaudit.listening,
        subaudit.extracting,
    ])

    for func in top_level_functions:
        with subtests.test('bound to a hook', name=func.__name__):
            assert isinstance(func.__self__, Hook)

    with subtests.test('all bound to the same one'):
        assert len({func.__self__ for func in top_level_functions}) == 1


@attrs.frozen
class ChurnCounts:
    """Parameters for a churn test. (Helper for test_usable_in_high_churn.)"""

    # pylint: disable=too-few-public-methods  # This is an attrs data class.

    listeners: int
    """Total number of listeners."""

    delta: int
    """Number of listeners unsubscribed and resubscribed per test iteration."""

    iterations: int
    """Number of test iterations."""


@pytest.mark.slow
def test_usable_in_high_churn(
    subtests: SubTests,
    hook: Hook,
    event: str,
    make_listeners: _MultiSupplier[_MockListener],
) -> None:
    """~1000 listeners with frequent subscribe/unsubscribe isn't too slow."""
    counts = ChurnCounts(listeners=1000, delta=100, iterations=100)
    all_listeners = make_listeners(count=counts.listeners)
    prng = random.Random(18140838203929040771)

    all_expected_observations: List[List[int]] = []
    all_observations: List[List[int]] = []
    observations: List[int] = []

    for number, listener in enumerate(all_listeners):
        listener.side_effect = functools.partial(observations.append, number)
        hook.subscribe(event, listener)

    attached = list(range(counts.listeners))

    with clock_timer.ClockLogger() as timer:
        for _ in range(counts.iterations):
            detached: List[int] = []

            for _ in range(counts.delta):
                number = attached.pop(prng.randrange(len(attached)))
                hook.unsubscribe(event, all_listeners[number])
                detached.append(number)

            while detached:
                number = detached.pop(prng.randrange(len(detached)))
                hook.subscribe(event, all_listeners[number])
                attached.append(number)

            all_expected_observations.append(attached[:])
            subaudit.audit(event)
            all_observations.append(observations[:])
            observations.clear()

    with subtests.test('listener calls in correct order'):
        assert all_observations == all_expected_observations

    with subtests.test('elapsed time not excessive'):
        elapsed = datetime.timedelta(seconds=timer.total_elapsed)
        assert elapsed <= datetime.timedelta(seconds=5)  # Usually much faster.


# FIXME: Retest some common cases with audit events from the standard library.
@pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason='Python has no standard audit events before 3.8.',
    raises=AssertionError,
    strict=True,
)
def test_can_listen_to_standard_events_for_input(
    monkeypatch: MonkeyPatch,
    tmp_path: pathlib.Path,
    any_hook: _AnyHook,
    make_listeners: _MultiSupplier[_MockListener],
) -> None:
    """
    We can listen to the builtins.input and builtins.input/result events.

    See https://docs.python.org/3/library/audit_events.html. We should be able
    to listen to any event listed there, but these tests only try a select few.
    """
    prompt = 'What... is the airspeed velocity of an unladen swallow? '
    result = 'What do you mean? An African or European swallow?'
    stdin_path = tmp_path / 'input.txt'
    stdin_path.write_text(result, encoding='utf-8')
    expected_calls = [call.listen_prompt(prompt), call.listen_result(result)]

    parent = Mock()  # To assert the relative order of calls to child mocks.
    parent.listen_prompt, parent.listen_result = make_listeners(2)

    with any_hook.listening('builtins.input', parent.listen_prompt):
        with any_hook.listening('builtins.input/result', parent.listen_result):
            # FIXME: If this works, extract arrange/cleanup to a fixture.
            old_stdin_fd = os.dup(0)
            os.close(0)
            try:
                with open(stdin_path, encoding='utf-8') as file:
                    if file.fileno() != _STDIN_FILENO:
                        raise RuntimeError(
                            f'got fd {file.fileno()}, need {_STDIN_FILENO}')
                    with monkeypatch.context() as context:
                        context.setattr('sys.stdin', file)
                        returned_result = input(prompt)
            finally:
                os.dup2(old_stdin_fd, _STDIN_FILENO)
                os.close(old_stdin_fd)

            if returned_result != result:
                raise RuntimeError(
                    f'got input {returned_result!r}, need {result!r}')

    assert parent.mock_calls == expected_calls


@pytest.mark.xfail(
    sys.version_info >= (3, 8),
    reason='Python 3.8 has PEP 578, so @skip_if_unavailable should not skip.',
    raises=pytest.fail.Exception,
    strict=True,
)
def test_skip_if_unavailable_skips_before_3_8() -> None:
    wrapped = Mock(wraps=lambda: None)
    wrapper = subaudit.skip_if_unavailable(wrapped)
    with pytest.raises(unittest.SkipTest):
        wrapper()


@pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason='Python < 3.8 lacks PEP 578, so @skip_if_unavailable should skip.',
    raises=AssertionError,
    strict=True,
)
def test_skip_if_unavailable_does_not_skip_since_3_8() -> None:
    wrapped = Mock(wraps=lambda: None)
    wrapper = subaudit.skip_if_unavailable(wrapped)
    wrapper()
    wrapped.assert_called_once_with()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
