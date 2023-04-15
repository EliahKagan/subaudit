"""Tests for the subaudit module."""

# TODO: Maybe split this into multiple modules.

import contextlib
import functools
import sys
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    overload,
)
# TODO: Find a way to hint like _Call and _CallList, yet respect encapsulation.
from unittest.mock import _Call, _CallList, Mock, call
import uuid

import attrs
import pytest
from pytest import FixtureRequest
from pytest_mock import MockerFixture
from pytest_subtests import SubTests
from typing_extensions import Protocol, Self, TypeVarTuple, Unpack

import subaudit
from subaudit import Hook

_Ts = TypeVarTuple('_Ts')
"""Type variable tuple for input types."""

_R = TypeVar('_R')
"""Type variable for output type."""


class _FakeError(Exception):
    """Fake exception for testing."""


@pytest.fixture(name='maybe_raise', params=[False, True])
def _maybe_raise(request: FixtureRequest) -> Callable[[], None]:
    """
    A function that, when called, either raises _FakeError or does nothing.

    This parameterized fixture multiplies tests that use it, covering both the
    raising and non-raising cases.
    """
    def maybe_raise_now() -> None:
        if request.param:
            raise _FakeError

    return maybe_raise_now


def _generate(supplier: Callable[[], _R]) -> Iterator[_R]:
    """Yield indefinitely many elements, each by calling the supplier."""
    while True:
        yield supplier()


def _make_hook() -> Hook:
    """Create a hook instance."""
    return Hook()


@pytest.fixture(name='hook')
def _hook() -> Hook:
    """Hook instance."""
    return _make_hook()


@pytest.fixture(name='some_hooks')
def _some_hooks() -> Iterator[Hook]:
    """Iterator that gives as many Hook instances as needed."""
    return _generate(_make_hook)


class _UnboundMethodMock(Mock, Generic[Unpack[_Ts], _R]):
    """A mock that is also a descriptor, to behave like a function."""

    def __call__(__mock, self: object, *args: Unpack[_Ts]) -> _R:
        """Call the mock. (Only overridden to have stricter type hints.)"""
        return super().__call__(self, *args)

    @overload
    def __get__(
        self, instance: None, owner: Optional[Type[object]] = None,
    ) -> Self: ...

    @overload
    def __get__(
        self, instance: Any, owner: Optional[Type[object]] = None,
    ) -> Callable[[Unpack[_Ts]], _R]: ...

    def __get__(self, instance, owner=None):
        """When accessed through an instance, produce a "bound method"."""
        if instance is None:
            return self
        return functools.partial(self, instance)


@attrs.frozen
class _DerivedHookFixture:
    """A new Hook subclass's mocked (un)subscribe methods and an instance."""

    subscribe_method: Mock
    """Mock of the unbound subscribe method."""

    unsubscribe_method: Mock
    """Mock of the unbound unsubscribe method."""

    instance: Hook
    """Instance of the Hook subclass."""


@pytest.fixture(name='derived_hook')
def _derived_hook() -> _DerivedHookFixture:
    """Make a new Hook subclass with subscribe and unsubscribe mocked."""
    subscribe_method = _UnboundMethodMock[str, Callable[..., None], None]()
    unsubscribe_method = _UnboundMethodMock[str, Callable[..., None], None]()

    class MockedSubscribeUnsubscribeHook(Hook):
        subscribe = subscribe_method
        unsubscribe = unsubscribe_method

    return _DerivedHookFixture(
        subscribe_method=subscribe_method,
        unsubscribe_method=unsubscribe_method,
        instance=MockedSubscribeUnsubscribeHook(),
    )


def _make_event() -> str:
    """Create a randomly generated fake event name."""
    return f'test-subaudit-{uuid.uuid4()}'


@pytest.fixture(name='event')
def _event() -> str:
    """Randomly generated fake event name."""
    return _make_event()


@pytest.fixture(name='some_events')
def _some_events() -> Iterator[str]:
    """Iterator that gives as many fake event names as needed."""
    return _generate(_make_event)


class _MockLike(Protocol):  # TODO: Drop any members that aren't needed.
    """Protocol for objects with assert_* methods and call spying we need."""

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

    __slots__ = ()

    def __call__(self, *__args: Any) -> None: ...

    @property
    def side_effect(self) -> Optional[Callable[..., Any]]: ...

    @side_effect.setter
    def side_effect(self, __value: Optional[Callable[..., Any]]) -> None: ...


def _make_listener() -> _MockListener:
    """Create a mock listener."""
    return Mock()


@pytest.fixture(name='listener')
def _listener() -> _MockListener:
    """Mock listener."""
    return _make_listener()


@pytest.fixture(name='some_listeners')
def _some_listeners() -> Iterator[_MockListener]:
    """Iterator that gives as many mock listeners as needed."""
    return _generate(_make_listener)


@pytest.fixture(name='equal_listeners', params=[2, 3, 5])
def _equal_listeners(request: FixtureRequest) -> List[_MockListener]:
    """List of listeners that are different objects but all equal."""
    group_key = object()

    def in_group(other: object) -> bool:
        return getattr(other, 'group_key', None) is group_key

    def make_mock() -> _MockListener:
        return Mock(
            __eq__=Mock(side_effect=in_group),
            __hash__=Mock(return_value=hash(group_key)),
            group_key=group_key,
        )

    return [make_mock() for _ in range(request.param)]


@attrs.frozen
class _Extract:
    """
    Auditing event arguments extracted by a custom extractor.

    The point of this type is to be separate from anything in, or used by, the
    code under test, so no excessively specific behavior wrongly passes tests
    of more general behavior.
    """

    args: Tuple[Any, ...]
    """Event arguments extracted in a test."""


class _MockExtractor(_MockLike, Protocol):
    """Protocol for an extractor that supports some of the Mock interface."""

    __slots__ = ()

    def __call__(self, *__args: Any) -> _Extract: ...


@pytest.fixture(name='extractor')
def _extractor() -> _MockExtractor:
    """Mock extractor. Returns a tuple of its arguments."""
    return Mock(side_effect=lambda *args: _Extract(args))


# FIXME: Change each skipif to xfail with a condition.
#        Use a raises argument where appropriate.


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Python 3.8+ has sys.audit",
)
def test_audit_is_sys_audit_since_3_8() -> None:
    assert subaudit.audit is sys.audit


@pytest.mark.skipif(
    sys.version_info >= (3, 8),
    reason="Python 3.8+ has sys.audit",
)
def test_audit_is_sysaudit_audit_before_3_8() -> None:
    import sysaudit  # type: ignore[import]
    assert subaudit.audit is sysaudit.audit


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Python 3.8+ has sys.addaudithook",
)
def test_addaudithook_is_sys_addaudithook_since_3_8() -> None:
    assert subaudit.addaudithook is sys.addaudithook


@pytest.mark.skipif(
    sys.version_info >= (3, 8),
    reason="Python 3.8+ has sys.addaudithook",
)
def test_addaudithook_is_sysaudit_addaudithook_before_3_8() -> None:
    import sysaudit  # type: ignore[import]
    assert subaudit.addaudithook is sysaudit.addaudithook


def test_subscribed_listener_observes_event(
    hook: Hook, event: str, listener: _MockListener,
) -> None:
    hook.subscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_called_once_with('event', 'a', 'b', 'c')


def test_unsubscribed_listener_does_not_observe_event(
    hook: Hook, event: str, listener: _MockListener,
) -> None:
    hook.subscribe(event, listener)
    hook.unsubscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_not_called()


def test_subscribed_listener_does_not_observe_other_event(
    hook: Hook, some_events: Iterator[str], listener: _MockListener,
) -> None:
    """Subscribing to one event doesn't observe other events."""
    event1, event2 = some_events
    hook.subscribe(event1, listener)
    subaudit.audit(event2, 'a', 'b', 'c')
    listener.assert_not_called()


def test_listener_can_subscribe_multiple_events(
    hook: Hook, some_events: Iterator[str], listener: _MockListener,
) -> None:
    event1, event2 = some_events
    hook.subscribe(event1, listener)
    hook.subscribe(event2, listener)
    subaudit.audit(event1, 'a', 'b', 'c')
    subaudit.audit(event2, 'd', 'e')
    assert listener.mock_calls == [call('a', 'b', 'c'), call('d', 'e')]


def test_listeners_called_in_subscribe_order(
    hook: Hook, event: str, some_listeners: Iterator[_MockListener],
) -> None:
    ordering: List[int] = []
    listener1, listener2, listener3 = some_listeners
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)
    listener3.side_effect = functools.partial(ordering.append, 3)

    hook.subscribe(event, listener1)
    hook.subscribe(event, listener2)
    hook.subscribe(event, listener3)
    subaudit.audit(event)

    assert ordering == [1, 2, 3]


def test_listeners_called_in_subscribe_order_after_others_unsubscribe(
    hook: Hook, event: str, some_listeners: Iterator[_MockListener],
) -> None:
    ordering: List[int] = []
    listener1, listener2, listener3, listener4 = some_listeners
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)
    listener3.side_effect = functools.partial(ordering.append, 3)
    listener3.side_effect = functools.partial(ordering.append, 4)

    hook.subscribe(event, listener1)
    hook.subscribe(event, listener2)
    hook.subscribe(event, listener3)
    hook.subscribe(event, listener4)
    hook.unsubscribe(event, listener1)
    hook.unsubscribe(event, listener3)
    subaudit.audit(event)

    assert ordering == [2, 4]


def test_listeners_called_in_new_order_after_resubscribe(
    hook: Hook, event: str, some_listeners: Iterator[_MockListener],
) -> None:
    ordering: List[int] = []
    listener1, listener2 = some_listeners
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)

    hook.subscribe(event, listener1)
    hook.subscribe(event, listener2)
    hook.unsubscribe(event, listener1)
    hook.subscribe(event, listener1)
    subaudit.audit(event)

    assert ordering == [2, 1]


def test_cannot_unsubscribe_if_never_subscribed(
    hook: Hook, event: str, listener: _MockListener,
) -> None:
    with pytest.raises(ValueError):
        hook.unsubscribe(event, listener)


def test_cannot_unsubscribe_if_no_longer_subscribed(
    hook: Hook, event: str, listener: _MockListener,
) -> None:
    hook.subscribe(event, listener)
    hook.unsubscribe(event, listener)
    with pytest.raises(ValueError):
        hook.unsubscribe(event, listener)


@pytest.mark.parametrize('count', [0, 2, 3, 10])
def test_listener_observes_event_as_many_times_as_subscribed(
    count: int, hook: Hook, event: str, listener: _MockListener,
) -> None:
    for _ in range(count):
        hook.subscribe(event, listener)
    subaudit.audit(event)
    assert listener.call_count == count


@pytest.mark.parametrize('count', [2, 3, 10])
def test_can_unsubscribe_as_many_times_as_subscribed(
    count: int, hook: Hook, event: str, listener: _MockListener,
) -> None:
    for _ in range(count):
        hook.subscribe(event, listener)
    try:
        for _ in range(count):
            hook.subscribe(event, listener)
    except ValueError as error:
        pytest.fail(
            f"Couldn't subscribe then unsubscribe {count} times: {error!r}")


@pytest.mark.parametrize('count', [2, 3, 10])
def test_cannot_unsubscribe_more_times_than_subscribed(
    count: int, hook: Hook, event: str, listener: _MockListener,
) -> None:
    for _ in range(count):
        hook.subscribe(event, listener)
    with pytest.raises(ValueError):
        for _ in range(count + 1):
            hook.subscribe(event, listener)


def test_unsubscribe_keeps_other_listener(
    hook: Hook, event: str, some_listeners: Iterator[_MockListener],
) -> None:
    """Unsubscribing one listener doesn't prevent another from observing."""
    listener1, listener2 = some_listeners
    hook.subscribe(event, listener1)
    hook.subscribe(event, listener2)
    hook.unsubscribe(event, listener1)
    subaudit.audit(event, 'a', 'b', 'c')
    listener2.assert_called_once_with('a', 'b', 'c')


def test_unsubscribe_removes_last_equal_listener(
    hook: Hook, event: str, equal_listeners: List[_MockListener],
) -> None:
    for listener in equal_listeners:
        hook.subscribe(event, listener)
    hook.unsubscribe(event, equal_listeners[0])
    subaudit.audit(event, 'a', 'b', 'c')
    equal_listeners[-1].assert_not_called()


def test_unsubscribe_keeps_non_last_equal_listeners(
    subtests: SubTests,
    hook: Hook,
    event: str,
    equal_listeners: List[_MockListener],
) -> None:
    """Unsubscribing removes no equal listeners besides the last subscribed."""
    for listener in equal_listeners:
        hook.subscribe(event, listener)
    hook.unsubscribe(event, equal_listeners[0])
    subaudit.audit(event, 'a', 'b', 'c')

    for index, listener in enumerate(equal_listeners[:-1]):
        with subtests.test(listener_index=index):
            listener.assert_called_once_with('a', 'b', 'c')


def test_cannot_unsubscribe_listener_from_other_hook(
    some_hooks: Iterator[Hook], event: str, listener: _MockListener,
) -> None:
    hook1, hook2 = some_hooks
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
    some_events: Iterator[str],
    some_listeners: Iterator[_MockListener],
) -> None:
    event1, event2 = some_events
    listener1, listener2 = some_listeners
    hook.subscribe(event1, listener1)
    mock = mocker.patch('subaudit.addaudithook')
    hook.subscribe(event2, listener2)
    mock.assert_not_called()


def test_second_instance_adds_audit_hook_on_first_subscribe(
    mocker: MockerFixture,
    some_hooks: Iterator[Hook],
    some_events: Iterator[str],
    some_listeners: Iterator[_MockListener],
) -> None:
    """Different Hook objects do not share the same audit hook."""
    hook1, hook2 = some_hooks
    event1, event2 = some_events
    listener1, listener2 = some_listeners
    hook1.subscribe(event1, listener1)
    mock = mocker.patch('subaudit.addaudithook')
    hook2.subscribe(event2, listener2)
    mock.assert_called_once()


def test_listening_does_not_observe_before_enter(
    hook: Hook, event: str, listener: _MockListener,
) -> None:
    """The call to listening does not itself subscribe."""
    context_manager = hook.listening(event, listener)
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
    hook: Hook, event: str, listener: _MockListener,
) -> None:
    """In the block of a with statement, the listener is subscribed."""
    with hook.listening(event, listener):
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
    hook: Hook,
    event: str,
    listener: _MockListener,
) -> None:
    """After exiting the with statement, the listener is not subscribed."""
    with contextlib.suppress(_FakeError):
        with hook.listening(event, listener):
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
    hook: Hook,
    event: str,
    listener: _MockListener,
) -> None:
    """The listening context manager works in (simple yet) nontrivial usage."""
    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with hook.listening(event, listener):
            subaudit.audit('d')
            subaudit.audit('e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert listener.mock_calls == [call('d'), call('e', 'f')]


def test_listening_enter_returns_none(
    hook: Hook, event: str, listener: _MockListener,
) -> None:
    """The listening context manager isn't meant to be used with "as"."""
    with hook.listening(event, listener) as context:
        pass
    assert context is None


def test_extracting_does_not_observe_before_enter(
    hook: Hook, event: str, extractor: _MockExtractor,
) -> None:
    context_manager = hook.extracting(event, extractor)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager:
        pass
    extractor.assert_not_called()


def test_extracting_does_not_extract_before_enter(
    hook: Hook, event: str, extractor: _MockExtractor,
) -> None:
    context_manager = hook.extracting(event, extractor)
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
    hook: Hook, event: str, extractor: _MockExtractor,
) -> None:
    with hook.extracting(event, extractor):
        subaudit.audit(event, 'a', 'b', 'c')
    extractor.assert_called_once_with('a', 'b', 'c')


def test_extracting_extracts_between_enter_and_exit(
    hook: Hook, event: str, extractor: _MockExtractor,
) -> None:
    with hook.extracting(event, extractor) as extracts:
        subaudit.audit(event, 'a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def text_extracting_enter_calls_subscribe_exactly_once(
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
            derived_hook.subscribe_method.calls[0].args
        )
        with subtests.test(argument_name='hook'):
            assert subscribe_hook is derived_hook.instance
        with subtests.test(argument_name='event'):
            assert subscribe_event == event


def test_extracting_enter_passes_appender_to_subscribe(
    derived_hook: _DerivedHookFixture, event: str, extractor: _MockExtractor,
) -> None:
    with derived_hook.instance.extracting(event, extractor) as extracts:
        subscribe_call, = derived_hook.subscribe_method.calls
        _, _, subscribe_listener = subscribe_call.args

    subscribe_listener('a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_does_not_observe_after_exit(
    maybe_raise: Callable[[], None],
    hook: Hook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor):
            maybe_raise()
    subaudit.audit(event, 'a', 'b', 'c')
    extractor.assert_not_called()


def test_extracting_does_not_extract_after_exit(
    maybe_raise: Callable[[], None],
    hook: Hook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    extracts: Optional[List[_Extract]] = None

    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor) as extracts:
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
        derived_hook.unsubscribe_method.calls[0].args
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

    unsubscribe_call, = derived_hook.unsubscribe_method.calls
    _, _, unsubscribe_listener = unsubscribe_call.args
    unsubscribe_listener('a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_observes_only_between_enter_and_exit(
    maybe_raise: Callable[[], None],
    hook: Hook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor):
            subaudit.audit('d')
            subaudit.audit('e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert extractor.mock_calls == [call('d'), call('e', 'f')]


def test_extracting_extracts_only_between_enter_and_exit(
    maybe_raise: Callable[[], None],
    hook: Hook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    extracts: Optional[List[_Extract]] = None

    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor) as extracts:
            subaudit.audit('d')
            subaudit.audit('e', 'f')
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


# FIXME: Test that high-throughput usage with ~300 listeners on the same event
#        remains fast.


# FIXME: Retest some common cases with audit events from the standard library.


# FIXME: Test the SyncHook class as well as the top-level stuff for the global
#        SyncHook instance. *Most* of this should be achieved by parametrizing
#        test fixtures that are already defined in this module.


# FIXME: Test skip_if_unavailable.


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
